/**
 * @file xmalloc.h
 * @brief Memory allocation and file stream wrappers with OOM checking for GEIF.
 */

#ifndef GEIF_XMALLOC_H
#define GEIF_XMALLOC_H

#include <stddef.h>
#include <stdio.h>

/**
 * @brief Allocates heap memory with out-of-memory abort and heap usage tracking.
 */
void *xmalloc(size_t n);

/**
 * @brief Allocates zero-initialized heap memory with out-of-memory abort.
 */
void *xcalloc(size_t count, size_t size);

/**
 * @brief Reallocates heap memory buffer with out-of-memory checking.
 */
void *xrealloc(void *ptr, size_t n);

/**
 * @brief Duplicates a string using xmalloc with null-termination and memory tracking.
 */
char *xstrdup(const char *s);

/**
 * @brief Frees memory buffer and decrements heap usage tracking counter.
 */
void  xfree(void *ptr);

/**
 * @brief Opens a file stream with transparent stdin/stdout support for "-".
 */
FILE *xfopen(const char *path, const char *mode);

/**
 * @brief Closes a file stream, safely flushing but not closing stdin/stdout/stderr.
 */
int   xfclose(FILE *fp);

/**
 * @brief Returns total active bytes currently tracked across all allocations.
 */
size_t xget_total_allocated(void);

/**
 * @brief Resets the active allocation counter back to 0.
 */
void   xreset_total_allocated(void);

#endif /* GEIF_XMALLOC_H */
