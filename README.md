# ZOLFOX Tools V1

`tools.zolfox.com` 第一版在线工具箱，当前先提供两个工具：

- 🪄 图片去水印：本地 LaMa CPU、Gemini、OpenAI，多种修复方式
- 📦 图片压缩：JPG / PNG / WebP，支持质量调整和输出格式选择

同时保留统一账号、游客体验额度、用户额度、手动充值和管理员后台，后续可以继续在同一个项目中增加 PDF、图片转换、尺寸调整等工具。

## 架构

```text
浏览器
  ↓
tools.zolfox.com
  ↓
宝塔 Nginx / HTTPS
  ↓
127.0.0.1:7860
  ↓
ZOLFOX Tools Docker
  ├── 图片去水印
  ├── 图片压缩
  ├── 账号 / 额度
  ├── 手动充值
  └── 管理后台
```

## 本地 Docker 测试

```bash
git clone https://github.com/walterkelly128-boop/Remove-watermark.git
cd Remove-watermark
cp .env.example .env
```

编辑 `.env`，至少设置：

```env
RC_API_KEY=你的RCouyi_API_KEY
RC_GEMINI_MODEL=你的Gemini图像模型
RC_OPENAI_MODEL=你的OpenAI图像模型
ADMIN_PASSWORD=一个强密码
GUEST_IP_SALT=一个长期稳定的随机密钥
```

然后：

```bash
docker compose build --no-cache
docker compose up -d
```

浏览器访问：

```text
http://127.0.0.1:7860
```

## 宝塔部署

项目目录例如：

```text
/www/wwwroot/tools.zolfox.com
```

进入目录：

```bash
cd /www/wwwroot/tools.zolfox.com
git pull
docker compose build
docker compose up -d
```

Compose 只绑定 `127.0.0.1:7860`，公网访问通过宝塔 Nginx 反向代理到 `127.0.0.1:7860`，再使用宝塔申请 SSL。

## 数据和磁盘

- `./data`：SQLite 账号、额度、充值和审计数据
- `./models`：本地 LaMa 模型
- 上传图片不写入业务数据库
- 压缩结果使用临时文件
- Docker 使用 CPU，不安装 CUDA

5GB 磁盘环境建议定期检查 Docker 镜像、构建缓存和日志，避免长期积累占满磁盘。

## 安全

不要把 `.env` 或真实 API Key 提交到 GitHub。

生产环境建议：

- 使用 HTTPS
- 设置强 `ADMIN_PASSWORD`
- 设置稳定的 `GUEST_IP_SALT`
- 配置管理员 TOTP 2FA
- 仅让可信反向代理使用 `TRUST_PROXY=1`
- 定期备份 `./data/accounts.db`

## 合法使用

仅处理你拥有或明确获得授权可以修改的图片，并遵守图片来源平台的版权、条款和署名要求。
