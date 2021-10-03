# Git用法（https://learngitbranching.js.org/?locale=zh_CN）

## ***<u>git 基本操作</u>***

### ==git主要命令==

#### git commit 

添加修改，提交一次

#### git branch （new image） 

创建名称为new image 的新分支

#### git  checkout newimage;git commit 

提交修改前切换到新分支上

#### git merge (new image)

将现在的分支和newimage分支合并

#### git rebase (main) 

在newimage节点上复制newimage并移动到main节点上，原来的分支不删

### ==git引用，指向==

#### git checkout (节点)

将head指向分离的节点

#### git checkout main^ 

指向main的上一级

#### git checkout main^num 

指向main的退num次，返回num次的父代

#### git branch -f main HEAD

把main移动到HEAD

#### git reset和git revert

撤销变更，reset是本地更改，revert是远程更改

### ==git移动提交记录==

#### git cherry-pick <提交号>

![image-20211001124526855](Git.assets/image-20211001124526855.png)

#### git rebase -i HEAD~4

![image-20211001125033875](Git.assets/image-20211001125033875.png)

### ==本地栈式提交==

#### git commit - - amend表示对当下分支进行更改

![image-20211002153100606](Git.assets/image-20211002153100606.png)

![image-20211002153110539](Git.assets/image-20211002153110539.png)

### ==标签==

#### git tag v1 c1

在c1上贴个标签，名字是v1。如果不具体写给谁加标签，会加到HEAD所在的节点处贴上v1的标签

#### git describe

![image-20211002164012788](Git.assets/image-20211002164012788.png)

![image-20211002164030958](Git.assets/image-20211002164030958.png)

![image-20211002164310804](Git.assets/image-20211002164310804.png)

### ==多分支==

#### 选择父提交项

![image-20211002182056334](Git.assets/image-20211002182056334.png)

![image-20211002182109360](Git.assets/image-20211002182109360.png)

![image-20211002182247308](Git.assets/image-20211002182247308.png)

![image-20211002182309209](Git.assets/image-20211002182309209.png)

（在这个游戏里可以用show solution看答案）

## ***<u>远程仓库</u>***

![image-20211002190553669](Git.assets/image-20211002190553669.png)

![image-20211002190634642](Git.assets/image-20211002190634642.png)

### ==远程分支==

![image-20211002190714371](Git.assets/image-20211002190714371.png)

![image-20211002190833461](Git.assets/image-20211002190833461.png)

![image-20211002190905669](Git.assets/image-20211002190905669.png)

### ==从远程仓库获取数据==

#### git fetch

![image-20211002191310079](Git.assets/image-20211002191310079.png)

![image-20211002192203200](Git.assets/image-20211002192203200.png)

![image-20211002192232181](Git.assets/image-20211002192232181.png)

![image-20211002192247988](Git.assets/image-20211002192247988.png)

#### git pull =git fetch+git merge

![image-20211002193324954](Git.assets/image-20211002193324954.png)

![image-20211002193538296](Git.assets/image-20211002193538296.png)

![image-20211002193559120](Git.assets/image-20211002193559120.png)

### ==模拟合作==

![image-20211002194235743](Git.assets/image-20211002194235743.png)

![image-20211002194247193](Git.assets/image-20211002194247193.png)

### ==从本地仓库上传==

#### git push

![image-20211002195328311](Git.assets/image-20211002195328311.png)

![image-20211002195413008](Git.assets/image-20211002195413008.png)

### ==历史偏离==

![image-20211002201336227](Git.assets/image-20211002201336227.png)

![image-20211002201404845](Git.assets/image-20211002201404845.png)

![image-20211002201427516](Git.assets/image-20211002201427516.png)

![image-20211002201616065](Git.assets/image-20211002201616065.png)

git merge 展示

![image-20211002201725786](Git.assets/image-20211002201725786.png)

![image-20211002201857268](Git.assets/image-20211002201857268.png)

![image-20211002201909731](Git.assets/image-20211002201909731.png)

#### git pull - - rebase 

![image-20211002201938143](Git.assets/image-20211002201938143.png)

![image-20211002202103948](Git.assets/image-20211002202103948.png)

### ==远程服务器拒绝remote rejected==

![image-20211002203029862](Git.assets/image-20211002203029862.png)

![image-20211002203052171](Git.assets/image-20211002203052171.png)

![image-20211002203104553](Git.assets/image-20211002203104553.png)

![image-20211002203834354](Git.assets/image-20211002203834354.png)

![image-20211002204042208](Git.assets/image-20211002204042208.png)

### ==git远程仓库高级操作==

