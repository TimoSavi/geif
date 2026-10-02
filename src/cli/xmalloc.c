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

/**
 * @brief Prints an out-of-memory error message to stderr and terminates process.
 *
 * @param[in] n Number of bytes that failed allocation.
 */
static void panic_oom(size_t n)
{
    fprintf(stderr, "geif: fatal: memory exhausted (failed to allocate %zu bytes)\n", n);
    exit(1);
}

/**
 * @brief Allocates heap memory with out-of-memory abort and heap usage tracking.
 *
 * @param[in] n Number of bytes to allocate (if 0, allocates 1 byte).
 * @return Pointer to allocated memory buffer.
 */
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

/**
 * @brief Allocates zero-initialized heap memory with out-of-memory abort.
 *
 * @param[in] count Number of elements.
 * @param[in] size  Size of each element.
 * @return Pointer to zero-initialized memory.
 */
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

/**
 * @brief Reallocates heap memory buffer with out-of-memory checking.
 *
 * @param[in] ptr Existing memory pointer (or NULL to malloc).
 * @param[in] n   New size in bytes.
 * @return Pointer to reallocated memory.
 */
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

/**
 * @brief Duplicates a string using xmalloc with null-termination and memory tracking.
 *
 * @param[in] s Source string to duplicate.
 * @return Newly allocated copy of string (or NULL if s is NULL).
 */
char *xstrdup(const char *s)
{
    if (!s) return NULL;
    size_t len = strlen(s);
    char *p = (char *)xmalloc(len + 1);
    memcpy(p, s, len + 1);
    return p;
}

/**
 * @brief Frees memory buffer and decrements heap usage tracking counter.
 *
 * @param[in] ptr Pointer to allocated memory to free.
 */
void xfree(void *ptr)
{
    if (!ptr) return;
#ifdef HAVE_MALLOC_USABLE_SIZE
    size_t sz = malloc_usable_size(ptr);
    if (total_allocation >= sz) total_allocation -= sz;
#endif
    free(ptr);
}

/**
 * @brief Opens a file stream with transparent stdin/stdout support for "-".
 *
 * Prevents multiple simultaneous open attempts on stdin/stdout, and emits
 * descriptive errno messages if fopen fails.
 *
 * @param[in] path File path or "-" for standard streams.
 * @param[in] mode Open mode ("r", "w", "a", etc.).
 * @return Opened FILE pointer, or NULL on error.
 */
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

/**
 * @brief Closes a file stream, safely flushing but not closing stdin/stdout/stderr.
 *
 * @param[in] fp FILE stream pointer to close.
 * @return 0 on success, or EOF on error.
 */
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

/**
 * @brief Returns total active bytes currently tracked across all allocations.
 *
 * @return Allocation total in bytes.
 */
size_t xget_total_allocated(void)
{
    return total_allocation;
}

/**
 * @brief Resets the active allocation counter back to 0.
 */
void xreset_total_allocated(void)
{
    total_allocation = 0;
}
