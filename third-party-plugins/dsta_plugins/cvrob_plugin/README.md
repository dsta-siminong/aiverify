# Project for cvrob_plugin plugin

For more information on AI Verify plugin developer, please refer to the [Developer Documentation](https://aiverify-foundation.github.io/aiverify-developer-tools/).

## Push project to GIT repo
1. Create a new blank GIT project.
2. Run the following commands to push the project to the GIT repo.

```
cd existing_folder
git init
git checkout -b "main"
git remote add origin <repo-url>
git add .
git commit -m "Initial commit"
git push -u origin main
```

## Create zip file for Plugin installation
Install the [aiverify-plugin](https://github.com/aiverify-foundation/aiverify-developer-tools/tree/main/aiverify-plugin) tool.

```
aiverify-plugin zip --pluginPath=<path to plugin directory>
```
