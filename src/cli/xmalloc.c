/**
 * @file xmalloc.c
 * @brief Memory allocation and stream wrappers with out-of-memory checking.
 *
 * Adapted from CEIF / GNU xmalloc with malloc_usable_size tracking
 * and transparent stdin/stdout handling.
 */

#include "xmalloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(__linux__) || defined(__GLIBC__)
#include <malloc.h>
#define HAVE_MALLOC_USABLE_SIZE 1
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#define malloc_usable_size(p) malloc_size(p)
#define HAVE_MALLOC_USABLE_SIZE 1
#endif

static size_t total_allocation = 0;
static int stdin_opened = 0;
static int stdout_opened = 0;

static void panic_oom(size_t n)
{
    fprintf(stderr, "geif: fatal: memory exhausted (failed to allocate %zu bytes)\n", n);
    exit(1);
}

void *xmalloc(size_t n)
{
    if (n == 0) n = 1;
    void *p = malloc(n);
    if (!p) panic_oom(n);

#ifdef HAVE_MALLOC_USABLE_SIZE
    total_allocation += malloc_usable_size(p);
#else
    total_allocation += n;
#endif

    return p;
}

void *xcalloc(size_t count, size_t size)
{
    if (count == 0 || size == 0) {
        count = 1;
        size = 1;
    }
    void *p = calloc(count, size);
    if (!p) panic_oom(count * size);

#ifdef HAVE_MALLOC_USABLE_SIZE
    total_allocation += malloc_usable_size(p);
#else
    total_allocation += count * size;
#endif

    return p;
}

void *xrealloc(void *ptr, size_t n)
{
    if (!ptr) return xmalloc(n);
    if (n == 0) n = 1;

#ifdef HAVE_MALLOC_USABLE_SIZE
    size_t old_sz = malloc_usable_size(ptr);
#endif

    void *p = realloc(ptr, n);
    if (!p) panic_oom(n);

#ifdef HAVE_MALLOC_USABLE_SIZE
    size_t new_sz = malloc_usable_size(p);
    if (new_sz > old_sz) total_allocation += (new_sz - old_sz);
    else if (total_allocation >= (old_sz - new_sz)) total_allocation -= (old_sz - new_sz);
#endif

    return p;
}

char *xstrdup(const char *s)
{
    if (!s) return NULL;
    size_t len = strlen(s);
    char *p = (char *)xmalloc(len + 1);
    memcpy(p, s, len + 1);
    return p;
}

void xfree(void *ptr)
{
    if (!ptr) return;
#ifdef HAVE_MALLOC_USABLE_SIZE
    size_t sz = malloc_usable_size(ptr);
    if (total_allocation >= sz) total_allocation -= sz;
#endif
    free(ptr);
}

FILE *xfopen(const char *path, const char *mode)
{
    if (!path || !mode) return NULL;

    if (strcmp(path, "-") == 0) {
        if (mode[0] == 'r') {
            if (stdin_opened) {
                fprintf(stderr, "geif: error: stdin already opened\n");
                return NULL;
            }
            stdin_opened = 1;
            return stdin;
        } else if (mode[0] == 'w' || mode[0] == 'a') {
            if (stdout_opened) {
                fprintf(stderr, "geif: error: stdout already opened\n");
                return NULL;
            }
            stdout_opened = 1;
            return stdout;
        }
    }

    FILE *fp = fopen(path, mode);
    if (!fp) {
        fprintf(stderr, "geif: error: cannot open '%s' (mode '%s'): %s\n",
                path, mode, strerror(errno));
        return NULL;
    }
    return fp;
}

int xfclose(FILE *fp)
{
    if (!fp) return 0;
    if (fp == stdin) {
        stdin_opened = 0;
        return 0;
    }
    if (fp == stdout) {
        fflush(stdout);
        stdout_opened = 0;
        return 0;
    }
    if (fp == stderr) {
        fflush(stderr);
        return 0;
    }
    return fclose(fp);
}

size_t xget_total_allocated(void)
{
    return total_allocation;
}

void xreset_total_allocated(void)
{
    total_allocation = 0;
}
