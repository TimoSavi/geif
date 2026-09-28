/**
 * @file xmalloc.h
 * @brief Memory allocation and file stream wrappers with OOM checking for GEIF.
 */

#ifndef GEIF_XMALLOC_H
#define GEIF_XMALLOC_H

#include <stddef.h>
#include <stdio.h>

void *xmalloc(size_t n);
void *xcalloc(size_t count, size_t size);
void *xrealloc(void *ptr, size_t n);
char *xstrdup(const char *s);
void  xfree(void *ptr);

FILE *xfopen(const char *path, const char *mode);
int   xfclose(FILE *fp);

size_t xget_total_allocated(void);
void   xreset_total_allocated(void);

#endif /* GEIF_XMALLOC_H */
