/* device-property replay: table embedded at build time, overridable at runtime with FK_PROPS */
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fkgsl.h"

#define FK_PROP_MAX 64

static const struct fk_prop *builtin = NULL;
static int builtin_count;

static struct fk_prop loaded[FK_PROP_MAX];
static int loaded_count;

#include "props_table.h"

static int hexnib(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int load_file(const char *path) {
  size_t text_len = 0;
  char *text = fk_read_file(path, &text_len);
  if (!text) {
    fk_log("FK_PROPS=%s cannot be opened: %s\n", path, strerror(errno));
    return -1;
  }
  char line[2048];
  int got = 0;
  size_t at = 0;
  while (at < text_len) {
    /* the same read-ahead the line reader had: a blob longer than the buffer continues on the next pass */
    size_t take = 0;
    while (take < sizeof line - 1 && at + take < text_len && text[at + take] != '\n') take++;
    memcpy(line, text + at, take);
    line[take] = '\0';
    at += take;
    if (at < text_len && text[at] == '\n') at++;
    char *cursor = line;
    while (isspace((unsigned char)*cursor)) cursor++;
    if (*cursor == '#' || *cursor == '\0' || *cursor == '\n') continue;
    char *type_text = strtok(cursor, " \t");
    char *value_text = NULL, *token;
    while ((token = strtok(NULL, " \t\n")) != NULL) {
      int allhex = 1;
      for (char *c = token; *c; c++)
        if (hexnib(*c) < 0) allhex = 0;
      if (allhex && (strlen(token) & 1) == 0 && strlen(token) >= 2) value_text = token;
    }
    if (!type_text || !value_text) continue;
    unsigned type = (unsigned)strtoul(type_text, NULL, 0);
    size_t nbytes = strlen(value_text) / 2;
    if (nbytes > sizeof ((struct fk_prop *)0)->bytes) nbytes = sizeof ((struct fk_prop *)0)->bytes;
    struct fk_prop *slot = NULL;
    for (int i = 0; i < loaded_count; i++)
      if (loaded[i].type == type) slot = &loaded[i];
    if (!slot) {
      if (loaded_count >= FK_PROP_MAX) break;
      slot = &loaded[loaded_count++];
      slot->type = type;
    }
    slot->len = nbytes;
    for (size_t i = 0; i < nbytes; i++) {
      int hi = hexnib(value_text[i * 2]), lo = hexnib(value_text[i * 2 + 1]);
      slot->bytes[i] = (unsigned char)((hi << 4) | lo);
    }
    got++;
  }
  free(text);
  fk_log("FK_PROPS=%s loaded %d properties\n", path, got);
  return got;
}

void fk_props_init(void) {
  const char *override;
  builtin = fk_prop_builtin;
  builtin_count = FK_PROP_BUILTIN_COUNT;
  override = getenv("FK_PROPS");
  if (override && *override) load_file(override);
}

static const struct fk_prop *find(unsigned type) {
  for (int i = loaded_count - 1; i >= 0; i--)
    if (loaded[i].type == type) return &loaded[i];
  for (int i = 0; i < builtin_count; i++)
    if (builtin[i].type == type) return &builtin[i];
  return NULL;
}

int fk_prop_fill(unsigned type, void *dst, size_t want) {
  const struct fk_prop *p = find(type);
  memset(dst, 0, want);
  if (!p) return 0;
  memcpy(dst, p->bytes, want < p->len ? want : p->len);
  return 1;
}

/* the payload returned for KGSL_PROP_DEVICE_SHADOW:
   { unsigned long gpuaddr @0; size_t size @8; unsigned int offset @16; }
   gpuaddr doubles as the mmap() offset; offset is the byte position of the timestamp in that page. */
int fk_prop_shadow(uint64_t *map_offset, uint32_t *ts_offset, uint64_t *size) {
  unsigned char raw[64];
  const char *env;
  uint64_t base = 0, bytes = 0x8000;
  uint32_t ts = 0x204;
  if (fk_prop_fill(0x02, raw, sizeof raw)) {
    memcpy(&base, raw, 8);
    memcpy(&bytes, raw + 8, 8);
    memcpy(&ts, raw + 16, 4);
  }
  if ((env = getenv("FK_SHADOW_OFFSET")) && *env) base = strtoull(env, NULL, 0);
  if ((env = getenv("FK_TS_OFFSET")) && *env) ts = (uint32_t)strtoul(env, NULL, 0);
  if ((env = getenv("FK_SHADOW_SIZE")) && *env) bytes = strtoull(env, NULL, 0);
  if (map_offset) *map_offset = base;
  if (ts_offset) *ts_offset = ts;
  if (size) *size = bytes;
  return base != 0;
}
