/* L6-4g Part 3: runs inside the sandbox. Forks /prog, waits, reports its CPU time, peak RSS and exit to fd 2. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
int main(void) {
    pid_t p = fork();
    if (p == 0) {
        int fd = open("/dev/null", O_WRONLY);
        dup2(fd, 2);
        execl("/prog", "prog", (char *)0);
        _exit(127);
    }
    int st; struct rusage ru;
    wait4(p, &st, 0, &ru);
    double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
    int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1, sig = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
    fprintf(stderr, "P3RUN cpu=%.4f rss_kib=%ld exit=%d signal=%d\n", cpu, ru.ru_maxrss, code, sig);
    return 0;
}
