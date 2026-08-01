#include <jni.h>
#include <android/log.h>
#include <pthread.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <string>
#include <atomic>
#include <errno.h>
#include <string.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "psiwatcher", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "psiwatcher", __VA_ARGS__)

static JavaVM* g_vm = nullptr;
static jclass g_cbClass = nullptr;           // com.example.llama.PsiWatcher$Callback
static jobject g_cbObj  = nullptr;           // GlobalRef to callback object
static jmethodID g_onEvt = nullptr;          // void onPsiEvent(int mode, int winUs, int thrUs)

static std::atomic<bool> g_running(false);
static pthread_t g_thread;

static int open_psi_fd(const char* path) {
    int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        // Some devices may not allow write access -> retry read-only (trigger setup fails in that case)
        fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    }
    return fd;
}

// mode: 0=some, 1=full
struct WatchArgs {
    int mode;
    int winUs;
    int thrUs;
};

static void* watch_thread(void* arg) {
    WatchArgs a = *(WatchArgs*)arg;
    delete (WatchArgs*)arg;

    const char* path = "/proc/pressure/memory";
    int fd = open_psi_fd(path);
    if (fd < 0) {
        LOGE("open(%s) failed: %s", path, strerror(errno));
        g_running = false;
        return nullptr;
    }

    // Set trigger: "some <win> <thr>\n" or "full <win> <thr>\n"
    // NOTE: may fail (EACCES) on devices that only allow R/O; poll then receives no events.
    {
        std::string cmd = (a.mode == 0 ? "some " : "full ");
        cmd += std::to_string(a.winUs);
        cmd += " ";
        cmd += std::to_string(a.thrUs);
        cmd += "\n";
        int wfd = open(path, O_WRONLY | O_CLOEXEC);
        if (wfd >= 0) {
            ssize_t n = write(wfd, cmd.c_str(), cmd.size());
            if (n < 0) {
                LOGE("write trigger failed: %s", strerror(errno));
            } else {
                LOGI("trigger set: %s", cmd.c_str());
            }
            close(wfd);
        } else {
            LOGE("open for write failed: %s", strerror(errno));
        }
    }

    // JVM attach
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm && g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            attached = true;
        }
    }

    struct pollfd pfd = { .fd = fd, .events = POLLPRI, .revents = 0 };

    while (g_running.load()) {
        int r = poll(&pfd, 1, -1); // block
        if (!g_running.load()) break;

        if (r > 0 && (pfd.revents & POLLPRI)) {
            // Consume the event: the file must be read for the next event to be delivered
            char buf[512];
            lseek(fd, 0, SEEK_SET);
            read(fd, buf, sizeof(buf));

            if (env && g_cbObj && g_onEvt) {
                env->CallVoidMethod(g_cbObj, g_onEvt, (jint)a.mode, (jint)a.winUs, (jint)a.thrUs);
            }
        } else if (r < 0 && errno != EINTR) {
            LOGE("poll error: %s", strerror(errno));
            break;
        }
    }

    if (attached) g_vm->DetachCurrentThread();
    close(fd);
    LOGI("watch thread exit");
    g_running = false;
    return nullptr;
}

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT jboolean JNICALL
Java_com_example_llama_PsiWatcher_nativeStart(
        JNIEnv* env, jclass,
        jobject callback, jint mode, jint windowUs, jint thresholdUs) {
    if (g_running.load()) return JNI_TRUE;

    // Callback class/method cache
    jclass cls = env->GetObjectClass(callback);
    if (!cls) return JNI_FALSE;
    g_cbClass = (jclass)env->NewGlobalRef(cls);
    g_cbObj   = env->NewGlobalRef(callback);
    g_onEvt   = env->GetMethodID(cls, "onPsiEvent", "(III)V");
    if (!g_onEvt) return JNI_FALSE;

    g_running = true;
    auto* args = new WatchArgs{ (int)mode, (int)windowUs, (int)thresholdUs };
    if (pthread_create(&g_thread, nullptr, watch_thread, args) != 0) {
        g_running = false;
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_example_llama_PsiWatcher_nativeStop(JNIEnv* env, jclass) {
    if (!g_running.load()) return;
    g_running = false;
    pthread_kill(g_thread, SIGUSR1); // wakeup aid (may not interrupt poll, but helps at shutdown)
    pthread_join(g_thread, nullptr);

    if (g_cbObj) { env->DeleteGlobalRef(g_cbObj); g_cbObj = nullptr; }
    if (g_cbClass) { env->DeleteGlobalRef(g_cbClass); g_cbClass = nullptr; }
    g_onEvt = nullptr;
}

} // extern "C"
