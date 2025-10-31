<!-- 7268fd34-a418-487e-a1df-b8a5a8336b38 53888e82-15b1-4d2a-97a8-1a8b1e73cee7 -->
# 计划：将代码上传到 GitHub

您的目标仓库是：`https://github.com/a527670091/IKE-GA`

下面是详细的步骤。您需要在您的项目根目录 `/data/chenminghao/code/IKE` 下的终端中，按顺序执行相应的命令。

### 步骤一：初始化本地 Git 仓库

首先，我们需要在您的项目文件夹中初始化一个 Git 仓库。这会在项目根目录下创建一个名为 `.git` 的隐藏文件夹，用来存放所有的版本管理信息。

```bash
git init
```

### 步骤二：将所有文件添加到暂存区

接下来，我们将项目中的所有文件添加到一个叫做“暂存区”的地方，准备进行提交。`.` 代表当前目录下的所有文件和文件夹。

```bash
git add .
```

### 步骤三：提交文件到本地仓库

现在，我们将暂存区的文件正式提交到本地仓库中，形成一个版本记录。`-m` 后面的 `"Initial commit"` 是本次提交的说明，意为“初始提交”，您可以根据需要修改引号内的内容。

```bash
git commit -m "Initial commit"
```

### 步骤四：关联到远程 GitHub 仓库

这一步是将您的本地仓库和您在 GitHub 上创建的空仓库关联起来。`origin` 是远程仓库的默认名称。

```bash
git remote add origin https://github.com/a527670091/IKE-GA.git
```

### 步骤五：重命名主分支为 main

为了与 GitHub 的当前默认分支名 `main` 保持一致，我们执行以下命令。这是一个推荐的做法。

```bash
git branch -M main
```

### 步骤六：推送代码到 GitHub

最后一步，就是将您本地仓库的代码推送到远程的 GitHub 仓库中。`-u` 参数会将本地的 `main` 分支和远程的 `main` 分支关联起来，这样以后推送就可以简化为 `git push` 命令。

```bash
git push -u origin main
```

执行完以上步骤后，刷新您的 GitHub 仓库页面，您应该就能看到您的代码了。

**注意：** 在执行 `git push` 时，终端可能会提示您输入 GitHub 的用户名和密码（或者是 Personal Access Token），请按照提示操作即可。