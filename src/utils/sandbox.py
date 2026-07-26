"""
sandbox.py — Native Linux seccomp policy loader for worker process sandboxing.

Integrates native Linux Secure Computing Mode (seccomp) filters using the
Berkeley Packet Filter (BPF) interface via ``prctl(2)``.  All filter
construction is self-contained — no external dependencies beyond the
standard library and the Linux kernel.

Policies are scoped to ingestion worker loops and block:
- Unauthorised filesystem write operations (``open`` / ``openat`` in
  ``O_WRONLY`` or ``O_RDWR`` mode, ``creat``, ``mkdir``, etc.)
- Keeping essential networking, memory, and signal syscalls available
  for designated RPC / REST fallback endpoints.

The module degrades gracefully on non-Linux platforms by compiling filters
but not applying them, so callers can unconditionally use the API in
cross-platform test suites.

Usage::

    from src.utils.sandbox import sandbox_policy, INGESTION_POLICY

    @sandbox_policy(INGESTION_POLICY)
    def run_ingestion_worker():
        # seccomp filter is active inside this function
        ...

    # Or as a context manager:
    with sandbox_policy(INGESTION_POLICY):
        run_ingestion_worker()

    # Apply explicitly (idempotent — only first call takes effect):
    sandbox_policy(INGESTION_POLICY).apply()
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import struct
import sys
from functools import wraps
from typing import Callable, FrozenSet, Optional, TypeVar

logger = logging.getLogger(__name__)

# ==========================================================================
# Platform detection
# ==========================================================================

_is_linux: bool = sys.platform == "linux"
_ARCH: str = platform.machine()

# Audit architecture constants (from <linux/audit.h>)
_AUDIT_ARCH_X86_64: int = 0xC000003E
_AUDIT_ARCH_AARCH64: int = 0xC00000B7
_AUDIT_ARCH_I386: int = 0x40000003

_ARCH_MAP: dict[str, int] = {
    "x86_64": _AUDIT_ARCH_X86_64,
    "amd64": _AUDIT_ARCH_X86_64,
    "aarch64": _AUDIT_ARCH_AARCH64,
    "arm64": _AUDIT_ARCH_AARCH64,
    "i386": _AUDIT_ARCH_I386,
    "i686": _AUDIT_ARCH_I386,
}

_CURRENT_AUDIT_ARCH: int = _ARCH_MAP.get(_ARCH, _AUDIT_ARCH_X86_64)

# ==========================================================================
# Seccomp constants
# ==========================================================================

# prctl(2) commands
_PR_SET_SECCOMP: int = 22
_PR_GET_SECCOMP: int = 23

# Seccomp modes
_SECCOMP_MODE_FILTER: int = 2

# Seccomp return actions (SECCOMP_RET_*)
_SECCOMP_RET_KILL_PROCESS: int = 0x80000000
_SECCOMP_RET_KILL_THREAD: int = 0x00000000
_SECCOMP_RET_ALLOW: int = 0x7FFF0000
_SECCOMP_RET_ERRNO: int = 0x00050000
_SECCOMP_RET_TRAP: int = 0x00030000  # Raises SIGSYS

# BPF instruction classes
_BPF_LD: int = 0x00
_BPF_JMP: int = 0x05
_BPF_RET: int = 0x06
_BPF_ALU: int = 0x04

# BPF sizes / modes
_BPF_W: int = 0x00
_BPF_ABS: int = 0x20
_BPF_K: int = 0x00

# BPF jump conditions (for reference; unlisted base values are unused)
_BPF_JEQ: int = 0x10

# BPF ALU ops
_BPF_AND: int = 0x50

# Composite BPF opcodes
_BPF_LD_W_ABS: int = _BPF_LD | _BPF_W | _BPF_ABS      # 0x20
_BPF_JMP_JEQ_K: int = _BPF_JMP | _BPF_JEQ | _BPF_K     # 0x15
_BPF_ALU_AND_K: int = _BPF_ALU | _BPF_AND | _BPF_K      # 0x04
_BPF_RET_K: int = _BPF_RET | _BPF_K                      # 0x06

# ==========================================================================
# System call numbers (Linux x86_64 ABI)
# ==========================================================================

class Syscall:
    """Canonical x86_64 Linux syscall numbers used by the seccomp filter."""
    READ: int = 0
    WRITE: int = 1
    OPEN: int = 2
    CLOSE: int = 3
    STAT: int = 4
    FSTAT: int = 5
    LSTAT: int = 6
    POLL: int = 7
    LSEEK: int = 8
    MMAP: int = 9
    MPROTECT: int = 10
    MUNMAP: int = 11
    BRK: int = 12
    RT_SIGACTION: int = 13
    RT_SIGPROCMASK: int = 14
    RT_SIGRETURN: int = 15
    IOCTL: int = 16
    PREAD64: int = 17
    PWRITE64: int = 18
    READV: int = 19
    WRITEV: int = 20
    ACCESS: int = 21
    PIPE: int = 22
    SELECT: int = 23
    SCHED_YIELD: int = 24
    MREMAP: int = 25
    MSYNC: int = 26
    MINCORE: int = 27
    MADVISE: int = 28
    SHMGET: int = 29
    SHMAT: int = 30
    SHMCTL: int = 31
    DUP: int = 32
    DUP2: int = 33
    PAUSE: int = 34
    NANOSLEEP: int = 35
    GETITIMER: int = 36
    ALARM: int = 37
    SETITIMER: int = 38
    GETPID: int = 39
    SENDFILE: int = 40
    SOCKET: int = 41
    CONNECT: int = 42
    ACCEPT: int = 43
    SENDTO: int = 44
    RECVFROM: int = 45
    SENDMSG: int = 46
    RECVMSG: int = 47
    SHUTDOWN: int = 48
    BIND: int = 49
    LISTEN: int = 50
    GETSOCKNAME: int = 51
    GETPEERNAME: int = 52
    SOCKETPAIR: int = 53
    SETSOCKOPT: int = 54
    GETSOCKOPT: int = 55
    CLONE: int = 56
    FORK: int = 57
    VFORK: int = 58
    EXECVE: int = 59
    EXIT: int = 60
    WAIT4: int = 61
    KILL: int = 62
    UNAME: int = 63
    SEMGET: int = 64
    SEMOP: int = 65
    SEMCTL: int = 66
    SHMDT: int = 67
    MSGGET: int = 68
    MSGSND: int = 69
    MSGRCV: int = 70
    MSGCTL: int = 71
    FCNTL: int = 72
    FLOCK: int = 73
    FSYNC: int = 74
    FDATASYNC: int = 75
    TRUNCATE: int = 76
    FTRUNCATE: int = 77
    GETDENTS: int = 78
    GETCWD: int = 79
    CHDIR: int = 80
    FCHDIR: int = 81
    RENAME: int = 82
    MKDIR: int = 83
    RMDIR: int = 84
    CREAT: int = 85
    LINK: int = 86
    UNLINK: int = 87
    SYMLINK: int = 88
    READLINK: int = 89
    CHMOD: int = 90
    FCHMOD: int = 91
    CHOWN: int = 92
    FCHOWN: int = 93
    LCHOWN: int = 94
    UMASK: int = 95
    GETTIMEOFDAY: int = 96
    GETRLIMIT: int = 97
    GETRUSAGE: int = 98
    SYSINFO: int = 99
    TIMES: int = 100
    PTRACE: int = 101
    GETUID: int = 102
    SYSLOG: int = 103
    GETGID: int = 104
    SETUID: int = 105
    SETGID: int = 106
    GETEUID: int = 107
    GETEGID: int = 108
    SETPGID: int = 109
    GETPPID: int = 110
    GETPGID: int = 111
    SETSID: int = 112
    SETREUID: int = 113
    SETREGID: int = 114
    GETGROUPS: int = 115
    SETGROUPS: int = 116
    SETRESUID: int = 117
    GETRESUID: int = 118
    SETRESGID: int = 119
    GETRESGID: int = 120
    GETPGROUP: int = 121
    SETFSUID: int = 122
    SETFSGID: int = 123
    GETSID: int = 124
    CAPGET: int = 125
    CAPSET: int = 126
    RT_SIGPENDING: int = 127
    RT_SIGTIMEDWAIT: int = 128
    RT_SIGQUEUEINFO: int = 129
    RT_SIGSUSPEND: int = 130
    SIGALTSTACK: int = 131
    UTIME: int = 132
    MKNOD: int = 133
    USELIB: int = 134
    PERSONALITY: int = 135
    USTAT: int = 136
    STATFS: int = 137
    FSTATFS: int = 138
    SYSFS: int = 139
    GETPRIORITY: int = 140
    SETPRIORITY: int = 141
    SCHED_SETPARAM: int = 142
    SCHED_GETPARAM: int = 143
    SCHED_SETSCHEDULER: int = 144
    SCHED_GETSCHEDULER: int = 145
    SCHED_GET_PRIORITY_MAX: int = 146
    SCHED_GET_PRIORITY_MIN: int = 147
    SCHED_RR_GET_INTERVAL: int = 148
    MLOCK: int = 149
    MUNLOCK: int = 150
    MLOCKALL: int = 151
    MUNLOCKALL: int = 152
    VHANGUP: int = 153
    MODIFY_LDT: int = 154
    PIVOT_ROOT: int = 155
    _SYSCTL: int = 156
    PRCTL: int = 157
    ARCH_PRCTL: int = 158
    ADJTIMEX: int = 159
    SETRLIMIT: int = 160
    CHROOT: int = 161
    SYNC: int = 162
    ACCT: int = 163
    SETTIMEOFDAY: int = 164
    MOUNT: int = 165
    UMOUNT2: int = 166
    SWAPON: int = 167
    SWAPOFF: int = 168
    REBOOT: int = 169
    SETHOSTNAME: int = 170
    SETDOMAINNAME: int = 171
    IOPL: int = 172
    IOPERM: int = 173
    CREATE_MODULE: int = 174
    INIT_MODULE: int = 175
    DELETE_MODULE: int = 176
    GET_KERNEL_SYMS: int = 177
    QUERY_MODULE: int = 178
    QUOTACTL: int = 179
    NFSSERVCTL: int = 180
    GETPMSG: int = 181
    PUTPMSG: int = 182
    AFS_SYSCALL: int = 183
    TUXCALL: int = 184
    SECURITY: int = 185
    GETTID: int = 186
    READAHEAD: int = 187
    SETXATTR: int = 188
    LSETXATTR: int = 189
    FSETXATTR: int = 190
    GETXATTR: int = 191
    LGETXATTR: int = 192
    FGETXATTR: int = 193
    LISTXATTR: int = 194
    LLISTXATTR: int = 195
    FLISTXATTR: int = 196
    REMOVEXATTR: int = 197
    LREMOVEXATTR: int = 198
    FREMOVEXATTR: int = 199
    TKILL: int = 200
    TIME: int = 201
    FUTEX: int = 202
    SCHED_SETAFFINITY: int = 203
    SCHED_GETAFFINITY: int = 204
    SET_THREAD_AREA: int = 205
    IO_SETUP: int = 206
    IO_DESTROY: int = 207
    IO_GETEVENTS: int = 208
    IO_SUBMIT: int = 209
    IO_CANCEL: int = 210
    GET_THREAD_AREA: int = 211
    LOOKUP_DCOOKIE: int = 212
    EPOLL_CREATE: int = 213
    EPOLL_CTL_OLD: int = 214
    EPOLL_WAIT_OLD: int = 215
    REMAP_FILE_PAGES: int = 216
    GETDENTS64: int = 217
    SET_TID_ADDRESS: int = 218
    RESTART_SYSCALL: int = 219
    SEMTIMEDOP: int = 220
    FADVISE64: int = 221
    TIMER_CREATE: int = 222
    TIMER_SETTIME: int = 223
    TIMER_GETTIME: int = 224
    TIMER_GETOVERRUN: int = 225
    TIMER_DELETE: int = 226
    CLOCK_SETTIME: int = 227
    CLOCK_GETTIME: int = 228
    CLOCK_GETRES: int = 229
    CLOCK_NANOSLEEP: int = 230
    EXIT_GROUP: int = 231
    EPOLL_WAIT: int = 232
    EPOLL_CTL: int = 233
    TGKILL: int = 234
    UTIMES: int = 235
    VSERVER: int = 236
    MBIND: int = 237
    SET_MEMPOLICY: int = 238
    GET_MEMPOLICY: int = 239
    MQ_OPEN: int = 240
    MQ_UNLINK: int = 241
    MQ_TIMEDSEND: int = 242
    MQ_TIMEDRECEIVE: int = 243
    MQ_NOTIFY: int = 244
    MQ_GETSETATTR: int = 245
    KEXEC_LOAD: int = 246
    WAITID: int = 247
    ADD_KEY: int = 248
    REQUEST_KEY: int = 249
    KEYCTL: int = 250
    IOPRIO_SET: int = 251
    IOPRIO_GET: int = 252
    INOTIFY_INIT: int = 253
    INOTIFY_ADD_WATCH: int = 254
    INOTIFY_RMWATCH: int = 255
    MIGRATE_PAGES: int = 256
    OPENAT: int = 257
    MKDIRAT: int = 258
    MKNODAT: int = 259
    FCHOWNAT: int = 260
    FUTIMESAT: int = 261
    NEWFSTATAT: int = 262
    UNLINKAT: int = 263
    RENAMEAT: int = 264
    LINKAT: int = 265
    SYMLINKAT: int = 266
    READLINKAT: int = 267
    FCHMODAT: int = 268
    FACCESSAT: int = 269
    PSELECT6: int = 270
    PPOLL: int = 271
    UNSHARE: int = 272
    SET_ROBUST_LIST: int = 273
    GET_ROBUST_LIST: int = 274
    SPLICE: int = 275
    TEE: int = 276
    SYNC_FILE_RANGE: int = 277
    VMSPLICE: int = 278
    MOVE_PAGES: int = 279
    UTIMENSAT: int = 280
    EPOLL_PWAIT: int = 281
    SIGNALFD: int = 282
    TIMERFD_CREATE: int = 283
    EVENTFD: int = 284
    FALLOCATE: int = 285
    TIMERFD_SETTIME: int = 286
    TIMERFD_GETTIME: int = 287
    ACCEPT4: int = 288
    SIGNALFD4: int = 289
    EVENTFD2: int = 290
    EPOLL_CREATE1: int = 291
    DUP3: int = 292
    PIPE2: int = 293
    INOTIFY_INIT1: int = 294
    PREADV: int = 295
    PWRITEV: int = 296
    RT_TGSIGQUEUEINFO: int = 297
    PERF_EVENT_OPEN: int = 298
    RECVMMSG: int = 299
    FANOTIFY_INIT: int = 300
    FANOTIFY_MARK: int = 301
    PRLIMIT64: int = 302
    NAME_TO_HANDLE_AT: int = 303
    OPEN_BY_HANDLE_AT: int = 304
    CLOCK_ADJTIME: int = 305
    SYNCFS: int = 306
    SENDMMSG: int = 307
    SETNS: int = 308
    GETNS: int = 309
    PROCESS_VM_READV: int = 310
    PROCESS_VM_WRITEV: int = 311
    KCMP: int = 312
    FINIT_MODULE: int = 313
    SCHED_SETATTR: int = 314
    SCHED_GETATTR: int = 315
    RENAMEAT2: int = 316
    SECCOMP: int = 317
    GETRANDOM: int = 318
    MEMFD_CREATE: int = 319
    KEXEC_FILE_LOAD: int = 320
    BPF: int = 321
    EXECVEAT: int = 322
    USERFAULTFD: int = 323
    MEMBARRIER: int = 324
    MLOCK2: int = 325
    COPY_FILE_RANGE: int = 326
    PREADV2: int = 327
    PWRITEV2: int = 328
    PKEY_MPROTECT: int = 329
    PKEY_ALLOC: int = 330
    PKEY_FREE: int = 331
    STATX: int = 332
    IO_PGETEVENTS: int = 333
    RSEQ: int = 334
    CLONE3: int = 435

# Linux open(2) flag masks
_O_ACCMODE: int = 0o3
_O_RDONLY: int = 0o0
_O_WRONLY: int = 0o1
_O_RDWR: int = 0o2

# ==========================================================================
# ctypes Seccomp / BPF Structures
# ==========================================================================


class SockFilter(ctypes.Structure):
    """A single BPF instruction — matches ``struct sock_filter``."""

    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    """BPF program container — matches ``struct sock_fprog``."""

    _fields_ = [
        ("len", ctypes.c_uint16),
        ("_pad", ctypes.c_uint16),                      # implicit padding
        ("filter", ctypes.POINTER(SockFilter)),
    ]


# ==========================================================================
# libc / prctl bindings
# ==========================================================================

try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _LIBC = None  # Non-Linux / unsupported

if _LIBC is not None:
    _LIBC.prctl.restype = ctypes.c_int
    _LIBC.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]


def _raw_prctl(option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> int:
    """Thin wrapper around ``prctl(2)`` from libc."""
    if _LIBC is None:
        raise OSError("libc not available — not a Linux target")
    result = _LIBC.prctl(option, arg2, arg3, arg4, arg5)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return result


# ==========================================================================
# Policy definition
# ==========================================================================

#: Essential syscalls required by CPython and threading runtime.
_CPYTHON_ESSENTIALS: FrozenSet[int] = frozenset([
    Syscall.READ,
    Syscall.WRITE,
    Syscall.CLOSE,
    Syscall.STAT,
    Syscall.FSTAT,
    Syscall.LSTAT,
    Syscall.POLL,
    Syscall.LSEEK,
    Syscall.MMAP,
    Syscall.MPROTECT,
    Syscall.MUNMAP,
    Syscall.BRK,
    Syscall.RT_SIGACTION,
    Syscall.RT_SIGPROCMASK,
    Syscall.RT_SIGRETURN,
    Syscall.IOCTL,
    Syscall.PREAD64,
    Syscall.PWRITE64,
    Syscall.READV,
    Syscall.WRITEV,
    Syscall.ACCESS,
    Syscall.PIPE,
    Syscall.PIPE2,
    Syscall.SELECT,
    Syscall.SCHED_YIELD,
    Syscall.MREMAP,
    Syscall.MSYNC,
    Syscall.MINCORE,
    Syscall.MADVISE,
    Syscall.DUP,
    Syscall.DUP2,
    Syscall.DUP3,
    Syscall.PAUSE,
    Syscall.NANOSLEEP,
    Syscall.GETPID,
    Syscall.GETTID,
    Syscall.FUTEX,
    Syscall.SCHED_SETAFFINITY,
    Syscall.SCHED_GETAFFINITY,
    Syscall.EXIT,
    Syscall.EXIT_GROUP,
    Syscall.CLONE,
    Syscall.CLONE3,
    Syscall.SET_TID_ADDRESS,
    Syscall.SET_ROBUST_LIST,
    Syscall.GET_ROBUST_LIST,
    Syscall.RESTART_SYSCALL,
    Syscall.RSEQ,
    Syscall.TGKILL,
    Syscall.RT_SIGPENDING,
    Syscall.RT_SIGTIMEDWAIT,
    Syscall.RT_SIGQUEUEINFO,
    Syscall.RT_SIGSUSPEND,
    Syscall.RT_TGSIGQUEUEINFO,
    Syscall.SIGALTSTACK,
    Syscall.ARCH_PRCTL,
    Syscall.PRCTL,
    Syscall.PRLIMIT64,
    Syscall.GETTIMEOFDAY,
    Syscall.CLOCK_GETTIME,
    Syscall.CLOCK_GETRES,
    Syscall.CLOCK_NANOSLEEP,
    Syscall.TIME,
    Syscall.TIMES,
    Syscall.GETITIMER,
    Syscall.SETITIMER,
    Syscall.GETRLIMIT,
    Syscall.GETRUSAGE,
    Syscall.SYSINFO,
    Syscall.UNAME,
    Syscall.GETUID,
    Syscall.GETGID,
    Syscall.GETEUID,
    Syscall.GETEGID,
    Syscall.GETPPID,
    Syscall.GETPGID,
    Syscall.GETPGROUP,
    Syscall.GETSID,
    Syscall.CAPGET,
    Syscall.FADVISE64,
    Syscall.FCNTL,
    Syscall.GETDENTS64,
    Syscall.GETCWD,
    Syscall.READLINK,
    Syscall.READLINKAT,
    Syscall.NEWFSTATAT,
    Syscall.FACCESSAT,
    Syscall.GETRANDOM,
    Syscall.MEMBARRIER,
    Syscall.PKEY_MPROTECT,
    Syscall.PKEY_ALLOC,
    Syscall.PKEY_FREE,
    Syscall.SET_THREAD_AREA,
    Syscall.EPOLL_CREATE1,
    Syscall.EPOLL_CTL,
    Syscall.EPOLL_WAIT,
    Syscall.EPOLL_PWAIT,
    Syscall.EVENTFD2,
])

#: Networking syscalls used by ingestion workers for RPC / REST fallback.
_NETWORK_SYSCALLS: FrozenSet[int] = frozenset([
    Syscall.SOCKET,
    Syscall.CONNECT,
    Syscall.ACCEPT,
    Syscall.ACCEPT4,
    Syscall.SENDTO,
    Syscall.RECVFROM,
    Syscall.SENDMSG,
    Syscall.RECVMSG,
    Syscall.SENDMMSG,
    Syscall.RECVMMSG,
    Syscall.SHUTDOWN,
    Syscall.BIND,
    Syscall.LISTEN,
    Syscall.GETSOCKNAME,
    Syscall.GETPEERNAME,
    Syscall.SETSOCKOPT,
    Syscall.GETSOCKOPT,
    Syscall.GETTIMEOFDAY,
])

#: Syscalls that are unconditionally blocked because they imply filesystem
#: writes, privilege escalation, or kernel state modification.
_BLOCKED_SYSCALLS: FrozenSet[int] = frozenset([
    Syscall.CREAT,
    Syscall.MKDIR,
    Syscall.MKDIRAT,
    Syscall.RMDIR,
    Syscall.RENAME,
    Syscall.RENAMEAT,
    Syscall.RENAMEAT2,
    Syscall.LINK,
    Syscall.LINKAT,
    Syscall.UNLINK,
    Syscall.UNLINKAT,
    Syscall.SYMLINK,
    Syscall.SYMLINKAT,
    Syscall.MKNOD,
    Syscall.MKNODAT,
    Syscall.CHMOD,
    Syscall.FCHMOD,
    Syscall.FCHMODAT,
    Syscall.CHOWN,
    Syscall.FCHOWN,
    Syscall.FCHOWNAT,
    Syscall.LCHOWN,
    Syscall.TRUNCATE,
    Syscall.FTRUNCATE,
    Syscall.FALLOCATE,
    Syscall.MOUNT,
    Syscall.UMOUNT2,
    Syscall.PIVOT_ROOT,
    Syscall.CHROOT,
    Syscall.SWAPON,
    Syscall.SWAPOFF,
    Syscall.INIT_MODULE,
    Syscall.FINIT_MODULE,
    Syscall.DELETE_MODULE,
    Syscall.CREATE_MODULE,
    Syscall.KEXEC_LOAD,
    Syscall.KEXEC_FILE_LOAD,
    Syscall.REBOOT,
    Syscall.SETHOSTNAME,
    Syscall.SETDOMAINNAME,
    Syscall.IOPL,
    Syscall.IOPERM,
    Syscall.MODIFY_LDT,
    Syscall.ACCT,
    Syscall.SETTIMEOFDAY,
    Syscall.ADJTIMEX,
    Syscall.CLOCK_SETTIME,
    Syscall.SETRLIMIT,
    Syscall.PTRACE,
    Syscall.PERSONALITY,
    Syscall.SYSLOG,
    Syscall.SETUID,
    Syscall.SETGID,
    Syscall.SETREUID,
    Syscall.SETREGID,
    Syscall.SETRESUID,
    Syscall.SETRESGID,
    Syscall.SETFSGID,
    Syscall.SETFSUID,
    Syscall.SETPGID,
    Syscall.SETSID,
    Syscall.SETGROUPS,
    Syscall.CAPSET,
    Syscall.QUERY_MODULE,
    Syscall.SETXATTR,
    Syscall.FSETXATTR,
    Syscall.LSETXATTR,
    Syscall.REMOVEXATTR,
    Syscall.FREMOVEXATTR,
    Syscall.LREMOVEXATTR,
    Syscall.SECURITY,
    Syscall.SETNS,
    Syscall.UNSHARE,
    Syscall.BPF,
    Syscall.PERF_EVENT_OPEN,
    Syscall.PROCESS_VM_WRITEV,
    Syscall.IO_SUBMIT,
    Syscall.IO_CANCEL,
])


# ==========================================================================
# BPF Filter Builder
# ==========================================================================


class BPFBuilder:
    """Construct a seccomp BPF filter program instruction by instruction.

    The generated bytecode is suitable for passing to ``prctl(PR_SET_SECCOMP,
    SECCOMP_MODE_FILTER, prog)``.

    Parameters
    ----------
    arch:
        The ``AUDIT_ARCH_*`` constant for the target architecture.
    """

    def __init__(self, arch: int = _CURRENT_AUDIT_ARCH) -> None:
        self._arch: int = arch
        self._instructions: list[SockFilter] = []
        self._finalized: bool = False

        # --- Preamble: architecture check ---
        # Load seccomp_data.arch (offset 4 bytes)
        self._emit(_BPF_LD_W_ABS, 0, 0, 4)
        # If arch matches, jump to next instruction; otherwise kill
        self._emit(_BPF_JMP_JEQ_K, 1, 0, arch)
        # Architecture mismatch: kill process
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS)
        # --- Load syscall number ---
        # Load seccomp_data.nr (offset 0 bytes)
        self._emit(_BPF_LD_W_ABS, 0, 0, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow_syscall(self, nr: int) -> BPFBuilder:
        """Allow *nr* unconditionally."""
        self._emit(_BPF_JMP_JEQ_K, 0, 1, nr)
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW)
        return self

    def allow_syscalls(self, numbers: FrozenSet[int]) -> BPFBuilder:
        """Allow every syscall in *numbers* unconditionally."""
        for nr in sorted(numbers):
            self.allow_syscall(nr)
        return self

    def block_syscall_with_errno(self, nr: int, errno_val: int = 1) -> BPFBuilder:
        """Block *nr* by returning ``errno_val``."""
        self._emit(_BPF_JMP_JEQ_K, 0, 1, nr)
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | (errno_val & 0xFFFF))
        return self

    def block_syscall_with_trap(self, nr: int) -> BPFBuilder:
        """Block *nr* by raising SIGSYS signal.
        
        This is the preferred method for security-critical blocking as it
        immediately raises a signal that can be caught and logged, making
        syscall violations visible to monitoring systems.
        """
        self._emit(_BPF_JMP_JEQ_K, 0, 1, nr)
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_TRAP)
        return self

    def allow_open_read_only(self) -> BPFBuilder:
        """Allow ``open(2)`` only when the flags argument specifies ``O_RDONLY``.

        Loads ``args[1]`` (offset 24 in seccomp_data), masks with
        ``O_ACCMODE``, and kills the process if the result is non-zero
        (meaning ``O_WRONLY`` or ``O_RDWR`` was requested).
        """
        return self._check_write_flag(Syscall.OPEN, 24)

    def allow_openat_read_only(self) -> BPFBuilder:
        """Allow ``openat(2)`` only when the flags argument specifies ``O_RDONLY``."""
        return self._check_write_flag(Syscall.OPENAT, 32)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> BPFBuilder:
        """Append the default-deny rule and mark the program as complete."""
        if not self._finalized:
            # Default: kill process for any unmatched syscall
            self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS)
            self._finalized = True
        return self

    def compile(self) -> bytes:
        """Return the assembled BPF bytecode.

        Automatically calls :meth:`finalize` if it hasn't been called yet.
        """
        if not self._finalized:
            self.finalize()

        buf = bytearray()
        for ins in self._instructions:
            buf += struct.pack("<HBBI", ins.code, ins.jt, ins.jf, ins.k)
        return bytes(buf)

    def filter_count(self) -> int:
        """Return the number of BPF instructions emitted so far."""
        return len(self._instructions)

    def _emit(self, code: int, jt: int, jf: int, k: int) -> None:
        self._instructions.append(SockFilter(code, jt, jf, k))

    # ------------------------------------------------------------------
    # Internal: flag inspection helpers
    # ------------------------------------------------------------------

    def _check_write_flag(self, syscall_nr: int, arg_offset: int) -> BPFBuilder:
        """Block a syscall when its flags argument indicates write intent.

        Flow:
        1. Check if nr == syscall_nr; if not, skip to after this block.
        2. Load the low 32 bits of the flags argument at *arg_offset*.
        3. AND with ``O_ACCMODE`` (3).
        4. If result == 0 (O_RDONLY): jump to ALLOW (ret = 0x7FFF0000).
        5. Else: KILL_PROCESS.

        Then continues with an explicit ALLOW at the end so the next
        ``allow_syscall`` / ``block_syscall_with_errno`` call can chain.

        The BPF instructions in this method use *relative* jump offsets,
        which are resolved inside :meth:`compile`.
        """
        # JEQ syscall_nr: if match go next (jt=1), else jump to end-of-block (jf)
        self._emit(_BPF_JMP_JEQ_K, 0, 1, syscall_nr)
        # Not this syscall — jump to after this block (placeholder — patched later)
        patch_idx = len(self._instructions) - 1

        # Load flags from arg_offset
        self._emit(_BPF_LD_W_ABS, 0, 0, arg_offset)
        # AND with O_ACCMODE
        self._emit(_BPF_ALU_AND_K, 0, 0, _O_ACCMODE)
        # If result == 0 (O_RDONLY), jump to ALLOW (jt=1); else KILL (jf=0 to next)
        self._emit(_BPF_JMP_JEQ_K, 1, 0, 0)
        # KILL
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS)
        # ALLOW
        self._emit(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW)

        # Patch the bypass jump to land right after ALLOW
        current_end = len(self._instructions) - 1
        self._instructions[patch_idx].jf = current_end - patch_idx

        return self


# ==========================================================================
# Pre-built policies
# ==========================================================================

def _build_ingestion_filter() -> bytes:
    """Build the standard ingestion-worker seccomp filter.

    Policy:
    - All CPython essential syscalls allowed.
    - Network syscalls allowed (socket, connect, sendto, recvfrom, etc.).
    - ``open(2)`` / ``openat(2)`` allowed ONLY with ``O_RDONLY`` — write
      modes (``O_WRONLY``, ``O_RDWR``) are killed.
    - ``creat(2)``, ``mkdir(2)``, ``unlink(2)``, and all other filesystem-
      mutation syscalls are blocked with SIGSYS.
    - Privilege-escalation and kernel-modification syscalls are blocked with SIGSYS.
    - Default action: kill process.
    """
    bpf = BPFBuilder(_CURRENT_AUDIT_ARCH)

    # 1. Allow CPython essentials
    bpf.allow_syscalls(_CPYTHON_ESSENTIALS)

    # 2. Allow networking (for designated REST / RPC endpoints)
    bpf.allow_syscalls(_NETWORK_SYSCALLS)

    # 3. open(2) — only O_RDONLY
    bpf.allow_open_read_only()

    # 4. openat(2) — only O_RDONLY
    bpf.allow_openat_read_only()

    # 5. Block unsafe syscalls with SIGSYS (raises signal for monitoring)
    for nr in sorted(_BLOCKED_SYSCALLS):
        bpf.block_syscall_with_trap(nr)

    return bpf.compile()


#: Pre-compiled BPF bytecode for ingestion worker sandboxing.
INGESTION_POLICY: bytes = _build_ingestion_filter()


# ==========================================================================
# Sandbox policy loader
# ==========================================================================


class SandboxPolicy:
    """A seccomp filter policy that can be applied to the calling process.

    On Linux, applies the BPF filter via ``prctl(PR_SET_SECCOMP,
    SECCOMP_MODE_FILTER)``.  On non-Linux platforms this is a no-op.

    ``prctl`` seccomp can only be called **once per thread** — subsequent
    calls fail.  This class tracks whether the filter has been applied and
    silently skips re-applications.
    """

    def __init__(self, bpf_bytes: bytes, name: str = "unnamed") -> None:
        self._bpf_bytes: bytes = bpf_bytes
        self._name: str = name
        self._applied: bool = False
        self._filter_array = None  # ctypes array stored to prevent GC

    @property
    def name(self) -> str:
        return self._name

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def is_linux(self) -> bool:
        return _is_linux

    def apply(self) -> bool:
        """Apply the seccomp filter to the current thread.

        Returns ``True`` if the filter was successfully applied (or had
        already been applied), ``False`` if the platform does not support
        seccomp.
        """
        if self._applied:
            return True

        if not _is_linux:
            logger.debug("SandboxPolicy[%s]: non-Linux platform — skipping seccomp", self._name)
            return False

        if _LIBC is None:
            logger.warning("SandboxPolicy[%s]: libc not loadable — skipping seccomp", self._name)
            return False

        try:
            self._do_apply()
            self._applied = True
            logger.info(
                "SandboxPolicy[%s]: seccomp filter applied (%d instructions)",
                self._name,
                len(self._bpf_bytes) // 8,
            )
            return True
        except OSError as exc:
            # errno 1 (EPERM) can mean seccomp is already active or no_new_privs
            logger.warning(
                "SandboxPolicy[%s]: prctl(PR_SET_SECCOMP) failed: %s",
                self._name,
                exc,
            )
            return False

    def _do_apply(self) -> None:
        """Assemble ``sock_fprog`` and invoke ``prctl``."""
        ins_size = ctypes.sizeof(SockFilter)
        count = len(self._bpf_bytes) // ins_size
        ArrayType = SockFilter * count

        # Store filter array as instance attribute to prevent GC while
        # the seccomp filter is active — the kernel references this memory.
        self._filter_array = ArrayType()
        buf = ctypes.create_string_buffer(self._bpf_bytes)
        ctypes.memmove(
            ctypes.addressof(self._filter_array),
            ctypes.addressof(buf),
            len(self._bpf_bytes),
        )

        prog = SockFprog()
        prog.len = count
        prog.filter = ctypes.cast(
            ctypes.pointer(self._filter_array),
            ctypes.POINTER(SockFilter),
        )

        _raw_prctl(
            _PR_SET_SECCOMP,
            _SECCOMP_MODE_FILTER,
            ctypes.addressof(prog),
            0,
            0,
        )

    def __repr__(self) -> str:
        applied = "applied" if self._applied else "pending"
        return f"<SandboxPolicy name={self._name!r} state={applied} platform={'linux' if _is_linux else sys.platform}>"


# ==========================================================================
# Environment Variable Whitelisting
# ==========================================================================


# Default whitelist of safe environment variables for subprocess execution
# These are explicitly allowed as they are necessary for normal operation
_DEFAULT_ENV_WHITELIST: FrozenSet[str] = frozenset([
    "PATH",                    # Executable search paths
    "HOME",                    # User home directory
    "USER",                    # Current username
    "LANG",                    # Locale settings
    "LC_ALL",                  # Locale override
    "LC_CTYPE",                # Character encoding
    "TERM",                    # Terminal type
    "TZ",                      # Timezone
    "PYTHONPATH",              # Python module search path (if needed)
    "PYTHONUNBUFFERED",        # Python stdout/stderr buffering
    "PYTHONIOENCODING",        # Python I/O encoding
])


def sanitize_environment(
    env: Optional[dict[str, str]] = None,
    whitelist: Optional[FrozenSet[str]] = None,
    allowlist: Optional[FrozenSet[str]] = None,
) -> dict[str, str]:
    """Return a sanitized environment dictionary containing only whitelisted variables.

    This function prevents passing un-sanitized parent process environment
    variables to worker subprocesses, which could expose private secrets.

    Parameters
    ----------
    env:
        Source environment dictionary. If None, uses ``os.environ``.
    whitelist:
        Set of environment variable names to allow. If None, uses the
        default safe whitelist. Note: ``allowlist`` is an alias for
        ``whitelist`` for inclusive terminology. If provided, this overrides
        the default whitelist.
    allowlist:
        Alias for ``whitelist``. If both are provided, they are merged.

    Returns
    -------
    dict[str, str]
        A new dictionary containing only the whitelisted environment variables.

    Raises
    ------
    TypeError
        If ``whitelist`` or ``allowlist`` contains non-string keys.

    Example::

        from src.utils.sandbox import sanitize_environment

        # Use default whitelist
        clean_env = sanitize_environment()

        # Custom whitelist (overrides default)
        clean_env = sanitize_environment(
            whitelist=frozenset(["PATH", "HOME", "CUSTOM_VAR"])
        )

        # Pass to subprocess
        subprocess.run(["command"], env=clean_env)
    """
    if env is None:
        env = dict(os.environ)

    # If custom whitelist is provided, use it instead of default
    # Otherwise use default whitelist
    if whitelist is not None:
        if not all(isinstance(k, str) for k in whitelist):
            raise TypeError("whitelist must contain only string keys")
        allowed_vars = set(whitelist)
    else:
        allowed_vars = set(_DEFAULT_ENV_WHITELIST)
    
    # Merge allowlist if provided
    if allowlist is not None:
        if not all(isinstance(k, str) for k in allowlist):
            raise TypeError("allowlist must contain only string keys")
        allowed_vars.update(allowlist)

    # Build sanitized environment
    sanitized = {}
    for key, value in env.items():
        if key in allowed_vars:
            sanitized[key] = value

    logger.debug(
        "sanitize_environment: allowed %d/%d variables (whitelist size=%d)",
        len(sanitized),
        len(env),
        len(allowed_vars),
    )

    return sanitized


def get_default_env_whitelist() -> FrozenSet[str]:
    """Return the default environment variable whitelist.

    Returns
    -------
    FrozenSet[str]
        The default set of allowed environment variable names.
    """
    return _DEFAULT_ENV_WHITELIST


# ==========================================================================
# Convenience factory functions
# ==========================================================================

_DEFAULT_POLICY: Optional[SandboxPolicy] = None


def sandbox_policy(
    bpf_bytes: bytes | None = None,
    *,
    name: str = "sandbox",
) -> SandboxPolicy:
    """Create (or retrieve) a :class:`SandboxPolicy` that can be used as a
    decorator or context manager.

    Parameters
    ----------
    bpf_bytes:
        Pre-compiled BPF bytecode (e.g. ``INGESTION_POLICY``).
        If *None*, returns the default ingestion policy singleton.
    name:
        Human-readable label for logging.

    Returns
    -------
    SandboxPolicy
        A reusable policy object.  Call ``.apply()`` to activate seccomp, or
        use it as a context manager / decorator.

    Usage::

        from src.utils.sandbox import sandbox_policy, INGESTION_POLICY

        @sandbox_policy(INGESTION_POLICY)
        def my_worker():
            ...

        with sandbox_policy(INGESTION_POLICY):
            ...
    """
    if bpf_bytes is None:
        global _DEFAULT_POLICY
        if _DEFAULT_POLICY is None:
            _DEFAULT_POLICY = SandboxPolicy(INGESTION_POLICY, name="ingestion-default")
        return _DEFAULT_POLICY

    return SandboxPolicy(bpf_bytes, name=name)


F = TypeVar("F", bound=Callable)


def seccomp_guard(policy: SandboxPolicy) -> Callable[[F], F]:
    """Decorator that applies *policy* before calling the wrapped function.

    Example::

        from src.utils.sandbox import seccomp_guard, sandbox_policy, INGESTION_POLICY

        @seccomp_guard(sandbox_policy(INGESTION_POLICY))
        def run_ingestion_worker():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            policy.apply()
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# Make SandboxPolicy directly usable as a context manager and decorator.
# We need to monkey-patch __enter__ / __exit__ and __call__ onto the class.

def _policy_context_enter(self: SandboxPolicy) -> SandboxPolicy:
    self.apply()
    return self


def _policy_context_exit(
    self: SandboxPolicy,
    exc_type: object,
    exc_val: object,
    exc_tb: object,
) -> None:
    # Seccomp filters are permanent per-thread — no teardown needed.
    return None


def _policy_call(self: SandboxPolicy, func: F) -> F:
    """Allow SandboxPolicy instances to be used as decorators."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        self.apply()
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


# Attach context-manager and decorator protocol to SandboxPolicy.
# Using explicit assignment rather than ABC registration to keep the class
# self-contained and dependency-free.
SandboxPolicy.__enter__ = _policy_context_enter  # type: ignore[assignment]
SandboxPolicy.__exit__ = _policy_context_exit    # type: ignore[assignment]
SandboxPolicy.__call__ = _policy_call            # type: ignore[assignment]


# ==========================================================================
# Utility: check if seccomp is supported
# ==========================================================================

def seccomp_available() -> bool:
    """Return ``True`` if the kernel supports seccomp BPF filtering."""
    if not _is_linux:
        return False
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("Seccomp:"):
                    return "2" in line or "3" in line
    except OSError:
        pass
    return False


def current_seccomp_mode() -> int:
    """Return the current seccomp mode for this process.

    Modes: 0=disabled, 1=strict, 2=filter.
    Returns -1 on error or if seccomp is unavailable.
    """
    if not _is_linux or _LIBC is None:
        return -1
    try:
        return _raw_prctl(_PR_GET_SECCOMP, 0, 0, 0, 0)
    except OSError:
        return -1


__all__ = [
    # Core API
    "SandboxPolicy",
    "sandbox_policy",
    "seccomp_guard",
    # Pre-built policies
    "INGESTION_POLICY",
    # Builder
    "BPFBuilder",
    # Syscall enumeration
    "Syscall",
    # Utilities
    "seccomp_available",
    "current_seccomp_mode",
]
