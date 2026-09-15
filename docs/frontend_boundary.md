# 纯前端产品与开发验证边界

MSV 产品页面不需要 Python 服务，不探测服务状态，也不发送同源 `/api/*` 请求。
Python CLI/API 保留为开发、取证及 Transformers 对账工具；已有 API 和磁盘缓存契约不变。

## 实现选择与参考

- [Netron 浏览器版](https://github.com/lutzroeder/netron#install)将浏览器查看器和 Python 工具分别提供。MSV 同样把网页的模型查看与开发验证入口分开，不向网页加入服务状态或隐藏的开发开关。
- [Transformers.js](https://github.com/huggingface/transformers.js#readme)展示了模型相关逻辑直接在浏览器运行、无需服务端的成熟路径。本次只参考产品边界，复用 MSV 的配置组网、Graph IR 和现有 Hub 元数据读取，不引入推理库或下载权重。
- [MDN webkitdirectory](https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/webkitdirectory)与[文件输入及取消事件](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/input/file)：使用浏览器原生目录选择、FileList、File.text 和已有 header slice 读取器。精确匹配 `config.json`，取消不改变模型，不上传本地文件。
- [MDN Fetch 错误处理](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)：分别处理非 2xx、JSON 解析和网络失败；显示失败模型源，不用 MSV 代理掩盖失败。HF 配置失败后的 ModelScope 直连兼容逻辑保留。
- [Playwright 网络控制](https://playwright.dev/docs/network)：浏览器用例同时监听与拦截同源 `/api/*`。仅拦根路径 API，不误拦开发服务器的 `/src/api/*.js` 模块；HF 自身的公开 `/api/models` 搜索正常使用远程夹具。
- [MDN Flex 收缩规则](https://developer.mozilla.org/en-US/docs/Web/CSS/flex-shrink)：目录导出验收发现当前自适应布局会把画布压缩到低于其内容高度、遮挡导出按钮。仅辅助面板打开时固定画布高度并允许页面随内容增长；使用正常鼠标点击验证，不用强制点击跳过遮挡。

## 回归边界

- 中英文 × 桌面/移动端：页面入口、模型选项、Revision、远程加载/搜索成功与失败、旧 local 链接、目录错误、header、导出。
- 本地目录通过真实 FileChooser 设置测试目录；取消覆盖标准 `cancel` 事件和空 FileList 的处理，不声称自动操作了操作系统取消按钮。
- 59 个内置模型保留桌面展开、图边和成本回归；移动端保留代表模型及全部纯前端边界用例。MiniMap 在窄屏按现有设计隐藏。
- 前端单元测试、模型验证、生产构建，以及独立 Python 验证工具测试共同检查变化。网络夹具证明浏览器控制流和结果，不代表公网模型源当时可达。

来源规则见 [source_contract.json](details/models/source_contract.json)：`frontend` 定义产品来源；原 `sources`、缓存及错误状态码字段继续约束 Python 工具。
