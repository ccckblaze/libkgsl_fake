# libkgsl_fake

[English](README.md) · **简体中文**

一个跑在 Linux/glibc 和 Android/bionic 上的 **LD_PRELOAD 垫片**：把高通 Adreno 的 KGSL 设备层
（`/dev/kgsl-3d0`）整段换成软件应答，让厂商那套闭源 OpenCL 编译器能在**一块没有 GPU 的 aarch64 机器上
离线编译，产出的 ISA 与真机逐位一致**。两代硬件都实测过：glibc 下的 a6xx，以及 Android 设备自己的
bionic 下的 a5xx。

先把三个"不是"说清楚，免得走错方向：

- **不是 `libgsl`/`libkgsl` 的替身**。它不实现任何用户态驱动 API，也不导出 `kgsl_*` 符号；拦的是
  `open`/`ioctl`/`mmap`/`dlsym` 这一层 libc 入口，厂商 runtime 仍是原厂那一份。
- **不是 GPU 模拟器**。一行指令都不仿真：kernel 提交只回执、不执行，凡是指望 GPU 回写的读数都不可信。
- **不是绕开厂商工具链的办法**。镜像里没有 `libOpenCL`/`libgsl`/`libllvm-qcom`，一个 ISA 字节也出不来。
  这里去掉的只有"必须先插着硬件"这一个前置条件。

一句话：**它把编译从执行里拆出来，让"这一步到底要不要 GPU"变成一条命令就能验证的事。**

## 为什么这条路走得通
Adreno 的 ISA 是在用户态由 `libllvm-qcom` 生成的，内核驱动在这条路上只干四件事：枚举设备、应答设备自述、
发内存句柄、发同步对象。真机录下来，**一次完整的 OpenCL 编译生命周期只用到 12 种 ioctl**（清单和字段含义见
`docs/findings.md` 第 1 节），而且一条 `GPU_COMMAND` 都不发。接口面小到这个份上，"照着实测回放"就比
"仿真硬件"便宜一到两个数量级——整个项目押的就是这一点，这也是它能成立的全部理由。

## 能力边界（都实测过）
| 档 | 内容 | 状态 |
|----|------|------|
| 一 | 截获并解析 `/dev/kgsl-3d0` 上的全部 ioctl 与命令批 | 已实现（`src/rec.c`，含 a6xx PM4 type-4/type-7 解码） |
| 二 | 无 GPU 环境跑通 `clCreateContext` + `clBuildProgram`，ISA 与真机逐位一致 | 已实现（`src/interpose.c` 等）：a6xx/glibc 与 a5xx/bionic 两代，整份产物的存盘重载跑得通，**没插卡也没板子的容器里整棵构建一张模型图**也跑得通（findings 第 8~9 节） |
| 三 | 仿真 a6xx PM4 + ISA6，把 kernel 真正**执行**出来 | 没做，也没打算做（理由见 `docs/findings.md` 第 3 节） |

要数值就找公开 CPU 后端（POCL / llvmpipe / `CPU=1`）取参考值；本垫片只对"代码半"负责。

## 它不做什么（也是实测）
下面三条看着像 bug，其实是边界：

- **这层复现代码，但不跑数。** 把编好的产物丢进垫片等它出数值，方向本身就错了：默认 `stub` 档位下 300 秒
  没有任何输出，只能强杀；`FK_EXEC=abort` 下 30 秒抛
  `Wait timeout: 30000 ms! (the signal is not set to 2, but 0)`。所以产物的验收只能落在真卡上，
  用 `tools/run_pkl_pair.py`。
- **会把缓冲存盘重建的构建，那些交给 GPU 去拷的缓冲什么也录不到。** a5xx 上，一部分 host→device 上传是
  runtime 内部的 CPU 拷贝，无卡也能原样到位；其余的是 GPU 命令，没有东西去执行，读回来就是零。一份产物
  实测：190 个权重缓冲里 101 个是对的，89 个是空的。数字并没有丢——它们就在模型文件里，而 kernel 要的布局
  也能从产物自己记录的形状推出来，所以正经做法是**重算**，而不是去找卡要。这层保证的仍然只有代码半。
- **启动维度只能来自卡，推不出来。** 有卡时，编译器会为每个 kernel 实测候选并固化选中的那一组线程数；
  无卡时计时恒为 0，选值就退化了。以 131 个 kernel 的监控模型实测：无卡产物里有 117 个 kernel 拿不到卡
  选过的线程数（87 个连本地组都没有，30 个落在 `local=1`）；真卡产物则是 35 个 `local=1`、30 个 global 带小数。
  同一张图放到真卡上跑，无卡产物 277 ms，真卡产物 12 ms。更要紧的是，自己凑一组卡没测过的维度不只是慢——
  实测 596 个输出里有 590 个对不上。所以合法取值只能从真卡产物里采：`tools/pkl_dims.py report` 看维度分布，
  `diff` 拿去跟真卡采的表对账。顺带一提，卡的读数本身也不确定：同一张卡编两次，131 组维度只有 73 组相同
  （与在跑的那份只有 75 组相同），可三份产物的输出逐位一致。自己核对一下就知道：`tools/pkl_dims.py diff`
  对真卡表是 131/131 全一致，对无卡产物是 44 一致、87 不一致。

其余数字和过程见 `docs/findings.md` 第 5~9 节（文档为英文）。

## 快速开始
```sh
make          # 产出 build/libkgsl_fake.so  build/libkgsl_rec.so  build/probe  build/tripwire
make test     # 六段断言，任何无 GPU 的 aarch64 Linux 都能跑

tools/fkgsl -- <命令>               # 注入假设备（默认 strict，GPU_COMMAND 只回执）
tools/fkgsl --abort-exec -- <命令>  # 收到 GPU_COMMAND 立即中止：证明这段流程不需要 GPU
tools/fkgsl --real -- <命令>        # 全部放行，用真 GPU 做对照
tools/fkgsl --rec -- <命令>         # 只录不骗（需要真 GPU）
```
macOS 上编不出来（glibc 版依赖 `open64`/`mmap64`/`RTLD_NEXT`，bionic 版依赖 bionic 自己的 libc），进容器跑：
```sh
docker run --rm --platform linux/arm64 -v "$PWD":/work -w /work <你的 arm64 镜像> make test
```
要出 Android/bionic 那一份，就在带着该平台自带 clang 的镜像里跑 `make bionic`（设备自己的用户态就够）：
产物是 `build/bionic/libkgsl_fake.so`，默认按代际取 a5xx 属性表，可用 `BIONIC_PROPS` 覆盖。对着这类设备
怎么录，见 `docs/record.md` 第 5 节。

要不要直接用一个装配好的 Android/bionic 盒子？配方其实很短：本仓跑 `make bionic`，再摆上一份录制下来的
设备属性表和一块板子的 `/sys` 身份，两样都放在同一个目录里。把这层**自己**的三个名字（`FK_PROPS`/
`FK_SYSFS`/`FK_TS_COVERAGE`）指过去，也不要替任何进程预加载。这些路径在一次构建里叫什么、要不要给某条
命令套上这层，那是构建自己的事，和本仓无关。

手上整台机器都没有 GPU？一条命令产出一份可用产物（代码半 + 数据半）：
```sh
tools/docker_gpless.sh <源码树> <model.onnx> [out.pkl]
# 容器里先构建垫片，再跑 model_codegen.py --cpu-exec，最后自检三条红线
# 结束时打印 GPU_COMMAND / UNHANDLED / TRIPWIRE 三个计数，必须全为 0
```
产物要拿到带目标 GPU 的机器上验收：`tools/run_pkl_pair.py <真卡产物.pkl> <你的.pkl>`（同一份新输入分别
回放两份产物，比对模型输出；它故意在没有卡的地方拒绝启动）、`tools/check_data_seed.py` 做缓冲级分类，
再用 `tools/pkl_dims.py report|diff` 看固化的启动维度、并跟真卡采的表对账。实测结果见
`docs/findings.md` 第 6~7 节。
最小可复现样本是 `build/probe`：枚举 → 建上下文 → 编译程序 → 落 ISA（路径由 `FK_ISA_OUT` 指定）。
只有 `probe full` 会走到提交工作，而那一步**就该**崩或卡住。

## 它是怎么工作的
### 1. 每个 libc 入口都要接住，一个都不能漏
`open` / `open64` / `openat` / `openat64` / `ioctl` / `mmap` / `mmap64` / `close` / `stat` / `lstat` /
`fstat` / `access` / `syscall()`。漏一个，调用方手里拿的就是**真的**设备 fd，然后带着一块真 GPU 生成一堆
"看着都对"的结果——这是最坏的失败方式。所以直连 `syscall(SYS_ioctl)` 那条路也接住了，`tests/tripwire.c`
就是它的回归测试。

要假装哪些节点由 `FK_KGSL_PATH` / `FK_ION_PATH` / `FK_PMSG_PATH` 决定（默认 `/dev/kgsl-3d0` 等）。
返回的"fd"是一个 memfd；后面所有分发都靠 fd → 类别这张表。

### 2. 设备自述回放：让编译器相信它就插在那张卡上
`DEVICE_GETPROPERTY` 的应答来自一张编译期内嵌的属性表，来源是 `props/a6xx.properties`——**从真卡上录下来的
原始小端字节**，每个 `KGSL_PROP_*` 一项（`DEVICE_INFO`、`DEVICE_SHADOW`、`UCHE_GMEM_VADDR`、`UCODE_VERSION`、
`HIGHEST_BANK_BIT`、`MIN_ACCESS_LENGTH`……）。`tools/genprops.py` 在 `make` 过程中把这份文件变成
`props_table.h`；`FK_PROPS=<文件>` 可以在运行期整表覆盖，不必重编。测试第 0 段就在两条路径之间对账，要求
ISA 完全一致。

代码生成的决策——寄存器分配、指令选择、bank 数量、特性开关——正是被这些字段驱动的。要支持另一张 GPU，
就重录一张表，别猜字段值。这里存了两代实测：`props/a6xx.properties`（10 次读）和 `props/a5xx.properties`
（5 次读，其中 3 个取值不同——包括 mmap 约定赖以建立的 `DEVICE_SHADOW`）。

### 3. `GPUOBJ` = 一块 memfd 分页切出去，并且照 KGSL 自己的 mmap 约定走
`GPUOBJ_ALLOC` 返回一个对象 id，`mmap(fd, len, …, offset)` 按内核的约定 **`offset = id << 12`** 找回它的
载体。载体是**同一块** memfd，按页对齐切片发出去、发出去就不再回收，于是两个对象不可能共享同一页。
以前是每个对象一块 memfd——真驱动可不会为每次分配占用调用方一个描述符——而在 bionic 设备上
`RLIMIT_NOFILE` 软限只有 1024（硬 4096），整棵构建的峰值存活对象数是 1005 个（分配 1120、释放 115），
于是 `memfd_create` 就在这个点上开始失败，`GPUOBJ_ALLOC` 报错，而 runtime 唯一的抱怨是重载刚才存下的
东西时 `clCreateBuffer` 返回了空指针。
`GPUOBJ_INFO` 故意不去另编一个 GPU 页表地址：a6xx 上 CPU 与 GPU 共享地址空间，runtime 直接把 `mmap`
返回的主机指针当 `gpuaddr` 用，所以把主机指针原样还给它才是自洽的答案。`DEVICE_SHADOW` 指的那一页要特殊
处理——见下一节。

### 4. 时间戳：给"等 GPU"一个终点
真 runtime 提交完工作就自旋在 `soptimestamp`/`eoptimestamp` 上。没有硬件回写就是永久挂住，看起来像垫片坏
了。垫片每回执一条 `GPU_COMMAND`，就把 shadow 页里那两个固定槽位（默认偏移 `0x204`，间距
`FK_TS_STRIDE=8`）往前推。**这东西的存在是为了让等待能结束，不是为了让学生结果正确。**
`FK_TS_COVERAGE=all`（把时间戳涂满每个映射区）在 a6xx 上原本只是排查用的重锤，在 a5xx 上是**硬性要求**：
a5xx runtime 自旋在这层叫不出名字的 shadow 字段上，不开它，整棵构建回执 12 条命令之后就会停在
`cl_a5x_wait_for_timestamp_thread` 里。

### 5. ION / dma-buf 的替身
`libgsl` 会向 `/dev/ion` 要 dma-buf。`src/ion.c` 用同一套 memfd 后端供这些地方；没有它，枚举阶段就死在
编译开始之前。

### 6. 改写 `dlsym`：`LD_PRELOAD` 的补集
最容易被忽略、也最吃重的一环。**`LD_PRELOAD` 只拦 PLT。** 像 `ctypes.CDLL("libc.so.6").mmap` 这种写法，
是先 `dlopen` 指定库、再在那个句柄上 `dlsym`，拿到的是 libc 自己的实现——符号插桩完全看不见。这个模式到处
都是：脚本层的 FFI、JNI，以及一切按指定句柄解析符号的加载器。所以垫片也接了 `dlsym`，并且**只在解析出来的
地址正好是 libc 那份实现时**才换成自己的函数。`RTLD_NEXT`/`RTLD_DEFAULT` 的查询原样放过——垫片自己就是靠
`RTLD_NEXT` 拿真实现的，改了它会自己套自己。
覆盖范围：`mmap/mmap64/ioctl/open*/stat*/fstat/access`；`tests/ctypes_lookup.py` 断言 11 个符号在两条路径上
落到同一个地址。

`dlvsym` 故意不接：从它那里取 `mmap` 拿到的是 libc 版本，越界访问时会当场炸响，而不是悄无声息滑到真硬件上
假装通过。

### 7. 严格模式与绊线：宁可中止，不可说谎
`FK_STRICT=1` 是默认值，两种情况直接中止：一个从没录过的 ioctl 号，以及一个不是垫片发出去的文件描述符上
收到了 KGSL ioctl（也就是调用方绕过了符号接管）。悄无声息用上真硬件是最坏的结果。`FK_EXEC=abort`
（`--abort-exec`）把"这一步到底要不要 GPU"变成一条命令就能给出的证据。

### 8. 录制器：先把接口面量清楚，再决定回放什么
`libkgsl_rec.so`（`src/rec.c`）只抄不骗：记录每个 ioctl 及其字段，并解码 a6xx 的 PM4 命令批
（type-4/type-7，`LOAD_STATE6` 可选择落盘）。"这段流程要不要 GPU"不用争论，在真机上跑一遍 `--rec`
就有答案。重录流程见 `docs/record.md`。

### 9. 板子的身份不是 GPU 问题
容器还缺两样东西，都不归这层管，但任何一样都会让闭源 runtime 在碰到设备节点之前就停下：

- **这台机器是哪颗 SoC。** 闭源 runtime 会先读 `/sys/devices/soc0/soc_id`（老一些的版本读
  `/sys/devices/system/soc/soc0/id`），愿意了才去找 GPU。没有这个节点，目标名就是 `UNKNOWN_SDM`，模型
  校验直接回绝 GPU 后端——`error_code=1006`，`No GPU device available for target UNKNOWN_SDM`——
  跟有没有设备垫片完全无关，因为它根本没被问到。`FK_SYSFS` 指定一个顶替 `/sys` 的目录，让录下来的节点
  替缺掉的节点回答。
- **系统对自己的描述。** API 24 以下的 Android/bionic 把全部系统属性放在**一个文件**
  `/dev/__properties__` 里，容器开局没有这个文件，于是每个属性都读成空。bionic 是以 `O_RDONLY` 打开、
  再按 `PROT_READ|PROT_WRITE` 映射它的——tmpfs 认这个组合，别的文件系统不认：所以容器必须**把文件拷进
  自己的 `/dev`**，直接把录制文件 bind mount 上去，映射会静默失败，看起来跟"这块板子没有属性"一模一样。

  不过要分清"编译依赖它"和"有人问平台问题时才需要它"：2026-11-03 在用本层组装出的盒子里两种都跑过，固定源
  编出来是同一份 2024 字节的 ISA，整模型也是 1045 个对象、19 段二进制、368 个 kernel，有属性区和没有
  一模一样（`tools/bundle_struct.py` 判定 soft pass：只有自动调优的 local work size 在动，那本来就是卡自己
  也不确定的东西，见 findings 第 7 节）。有属性区时盒子还会答 `platform=msm8996`、`model=OnePlus 3`，没有
  就都读成空。所以**公开 CI 要的只有 `/sys` 镜像，不需要属性区**——这也是为什么发出去的镜像带镜像、不带
  属性区：那 131,072 字节里有串号和三张网卡的地址。真要问平台问题的，仍然得自己交一份、并且是拷进去而不是挂载。

这两样都从一块能干活的板子上录，一条命令，不编造任何东西：`tools/record_board.sh <host> <目录>` 写出镜像、
取走属性区，并把容器要用它们的两行命令打印出来。

有个坑值得先说明，不然会白折腾一小时：静态链接的读者（`toybox cat`）看不见 `LD_PRELOAD`，所以镜像装好之后
`cat /sys/devices/soc0/soc_id` 仍然报 *no such file*，而所有动态链接的读者读到的都是镜像里的值。验证镜像请
用动态链接的程序，别用 `cat`。

## `make test` 到底在断言什么
| 段 | 断言 |
|----|------|
| 0 | 内嵌属性表与 `FK_PROPS` 运行期覆盖，两条路径产出的 ISA 必须逐位一致 |
| 1 | 编译段红线：0 条 `GPU_COMMAND`、0 条未录 ioctl、0 次真设备泄漏，且命中 `tests/golden/` 基线 |
| 2 | 执行段：**崩了或卡住才算通过**（真算对了，说明偷偷用了真卡） |
| 3 | `tripwire`：绕过符号接管的真设备 fd 必须被 strict 抓住（没有真卡时跳过） |
| 4 | 真实 kernel `tests/kernels/e2x.cl` 无 GPU 也能编译，4220 字节，与真机基线逐位一致 |
| 5 | `ctypes_lookup.py`：11 个接管符号在 ctypes/`dlsym` 路径上同样落在垫片实现里 |

另有 `tests/qcom_compiler.py`：把 tinygrad 的 QCOM 编译器这条真实 codegen 流水线整条跑在假设备上，
与固定探针、真卡三方对齐。基线由 `tests/golden.sh update` 在真机上建立，之后任何改动只要让 ISA 漂移，
这一套就会变红。

## 环境变量
| 变量 | 默认 | 作用 |
|------|------|------|
| `FK_VERBOSE` | 0 | 写日志 |
| `FK_LOG` | `/tmp/libkgsl_fake.log` | 日志落点 |
| `FK_STRICT` | 1 | 未录到的 ioctl / 真设备 fd 泄漏 → 中止 |
| `FK_EXEC` | `stub` | `stub` 只回执 timestamp；`abort` 收到 `GPU_COMMAND` 立即中止 |
| `FK_DISABLE` | 未设置 | 非空则全量放行（真 GPU 对照） |
| `FK_PROPS` | 未设置 | 用别的属性表覆盖内嵌的设备自述 |
| `FK_SYSFS` | 未设置 | 顶替 `/sys` 的目录：没有板子的容器缺的那几个身份节点由这里回答 |
| `FK_KGSL_PATH` / `FK_ION_PATH` / `FK_PMSG_PATH` | `/dev/kgsl-3d0` 等 | 接管哪些设备节点 |
| `FK_SHADOW_OFFSET` / `FK_TS_OFFSET` / `FK_SHADOW_SIZE` | 取自 `DEVICE_SHADOW` | 覆盖内存映射约定 |
| `FK_TS_STRIDE` | 8 | shadow 页两个时间戳槽位的间距 |
| `FK_TS_COVERAGE` | `shadow` | `all`＝涂满每个映射区；a5xx 必需，a6xx 上只是排查重锤 |
| `FK_REC_OUT` / `FK_DUMP_DIR` | — | 录制器日志 / PM4 落盘目录 |
| `FK_ISA_OUT` | `/tmp/fkgsl_probe_isa.bin` | `build/probe` 写 ISA 的位置 |
| `FK_FAKE_GPUADDR` | 0 | 覆盖 `GPUOBJ_INFO` 返回的 gpuaddr |

> 注：`tools/cpu_image_shim.py` 是"随模型工具链走的那份垫片"的第二份拷贝。
> 两边都守同一条规矩：实测而不是猜。真出现分叉，要改的是构建实际用到的那一份；这里留一份，
> 只为让本仓的实验脚本能独立跑起来。

## 目录结构
```
src/interpose.c  接管面：open*/ioctl/mmap*/close/stat*/access/syscall/dlsym
src/device.c     KGSL ioctl 语义（属性、GPUOBJ、上下文、同步、GPU_COMMAND）
src/props.c      设备自述回放 + FK_PROPS 覆盖
src/raw.c        裸系统调用：这层要用的那几个 libc 入口，不向加载器开口
src/mem.c        切片的 memfd 后端 + shadow 页时间戳
src/ion.c        ION 顶替
src/rec.c        录制器（含 a6xx type-4/type-7 PM4 解码）
include/         裁剪过的 KGSL UAPI 头，只留用到的结构体
props/           录自真机的设备自述（a6xx、a5xx），编译期默认值的唯一来源；
                 a5xx-sys/ 是同一块板子的 SoC 身份镜像（见第 9 节）
tests/           分阶段探针、泄漏守卫、ctypes 回归、基线哈希
docs/            findings.md 逐条实测结论；record.md 重录流程（均为英文）
tools/           fkgsl 包装脚本、属性表/PM4 生成器、record_board.sh（录板级身份）、
                 docker_gpless.sh（离线出包）、离线 codegen 实验，以及四个验收脚本
                 （run_pkl_pair / check_data_seed / pkl_dims / bundle_struct）
```

## 使用前提
- aarch64 上的 Linux/glibc 或 Android/bionic，并且装好**目标 GPU 那代**的厂商 runtime
  （`libOpenCL`、`libgsl`、`libllvm-qcom`）——垫片不替代其中任何一个。
- 没有板子的容器，除了设备垫片还需要板子的身份：一份 `/sys` 镜像（`FK_SYSFS`），以及 API 24 以下必须拷进
  自己 tmpfs `/dev` 的属性区。一条命令两样都能录下来——`tools/record_board.sh`。a5xx 那份镜像已经随仓库
  发布在 `props/a5xx-sys`；属性区不发，因为那个文件里带着这块板子的序列号和无线网卡地址——请从你自己
  可以拷贝的板子上录。
- x86 宿主得靠 QEMU 或原生 arm runner，跨架构不是本项目要解决的问题。
- 换了 GPU 代际、或升级了闭源 runtime，就重录属性表和 ioctl 面（`docs/record.md`），别猜字段。
- 只有编译段可信。任何"跑一下看看数值"的用法都在保证范围之外。

## 关于名字
`libkgsl_fake` 这个名字比项目实际做的事窄了：`libkgsl` 容易被读成"顶替厂商 UMD"（它不是），
`fake` 也只描述了"骗"这一半，没包括录制和离线出包。目录名先留着——牵连文档、脚本、容器路径太多。
准确的说法是**KGSL 设备垫片**：接管设备节点和 ioctl/mmap/dlsym 面，把厂商编译器的硬件前置条件拿掉。
真正会敲的命令是 `fkgsl`，那个名字没有歧义。

## 许可与一句提醒
本仓库不含任何密钥和业务数据：`props/` 里只有从公开设备节点读到的原始属性字节。
但它演示的这件事值得注意——设备存在性检查和 ioctl 应答，是可以整段用软件顶掉的，属性表本身也不例外。
如果把信任锚在"我能读到设备应答"上，这份代码就是一份现成的攻击样板：能验签的东西，验不了"是谁在回答"。
