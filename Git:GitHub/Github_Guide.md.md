# GitHub Notes

## What is GitHub?

GitHub is a web-based platform for version control and collaboration using Git. It allows developers to host, review, and manage code repositories.

## Basic Concepts

### Repository (Repo)

- A project folder that contains all your project files
- Can be public (visible to everyone) or private (only you/your team can see)

### Commit

- A snapshot of your code at a specific point in time
- Each commit has a message describing what changed

### Branch

- A parallel version of your repository
- Allows you to work on features without affecting the main code
- Main branch is typically called `main` or `master`

### Pull Request (PR)

- A way to propose changes to a repository
- Allows others to review your code before merging

## Common Git Commands

### Basic Setup

#### `git init`

**Explanation:** Initializes a new Git repository in the current directory. Creates a hidden `.git` folder that stores all version control information. Use this when starting a new project that you want to track with Git.

**Example:**

```bash
git init
```

#### `git clone <url>`

**Explanation:** Creates a copy of an existing repository from a remote location (like GitHub) to your local machine. Downloads all files, commit history, and branches. The URL can be HTTPS or SSH.

**Example:**

```bash
git clone https://github.com/username/repository.git
```

#### `git config --global user.name "Your Name"`

**Explanation:** Sets your name for all Git repositories on your computer. This name will appear in commit messages. The `--global` flag applies this setting to all repositories. Use your real name or preferred identifier.

**Example:**

```bash
git config --global user.name "John Doe"
```

#### `git config --global user.email "your.email@example.com"`

**Explanation:** Sets your email address for all Git repositories. This email will be associated with your commits. Should match the email used in your GitHub account.

**Example:**

```bash
git config --global user.email "john.doe@example.com"
```

### Making Changes

#### `git status`

**Explanation:** Shows the current state of your working directory. Displays which files are modified, staged, or untracked. Helps you see what changes are ready to be committed and what still needs attention.

**Example:**

```bash
git status
```

#### `git add <file>`

**Explanation:** Stages a specific file for commit. Moves the file from "modified" to "staged" state. Staged files are ready to be committed. You can add multiple files by listing them or use patterns.

**Example:**

```bash
git add index.html
git add src/components/Header.js
```

#### `git add .`

**Explanation:** Stages all changes in the current directory and subdirectories. The dot (`.`) means "current directory". This includes new files, modified files, and deleted files. Be careful - this stages everything, so review with `git status` first.

**Example:**

```bash
git add .
```

#### `git commit -m "message"`

**Explanation:** Creates a snapshot of your staged changes with a descriptive message. The `-m` flag allows you to write the commit message directly in the command. Each commit has a unique ID and represents a point in your project's history.

**Example:**

```bash
git commit -m "Add user authentication feature"
```

#### `git push`

**Explanation:** Uploads your local commits to the remote repository (like GitHub). Sends your changes to the server so others can see them. Usually pushes to the branch you're currently on. First push of a new branch may require: `git push -u origin branch-name`.

**Example:**

```bash
git push
git push origin main  # Push to specific branch
```

#### `git pull`

**Explanation:** Downloads and merges changes from the remote repository into your local repository. Combines `git fetch` (download) and `git merge` (integrate) in one command. Use this to get the latest changes from your team.

**Example:**

```bash
git pull
git pull origin main  # Pull from specific branch
```

### Branching

#### `git branch`

**Explanation:** Lists all local branches in your repository. The current branch is marked with an asterisk (\*). Shows you all available branches and which one you're currently working on.

**Example:**

```bash
git branch
```

#### `git branch <name>`

**Explanation:** Creates a new branch with the specified name. The new branch starts from your current branch's commit. Does not switch to the new branch - you remain on your current branch.

**Example:**

```bash
git branch feature/login
```

### Viewing History

#### `git log`

**Explanation:** Displays the commit history for the current branch. Shows commit hash, author, date, and commit message. Press `q` to exit. Useful for seeing what changes were made and when.

**Example:**

```bash
git log
```

#### `git log --oneline`

**Explanation:** Shows a compact, one-line version of the commit history. Each commit appears as a single line with its hash and message. Easier to scan through many commits quickly.

**Example:**

```bash
git log --oneline
```

#### `git diff`

**Explanation:** Shows the differences between your working directory and the staging area. Displays what has changed but not yet staged. Use `git diff --staged` to see differences between staged changes and the last commit.

**Example:**

```bash
git diff              # Unstaged changes
git diff --staged     # Staged changes
```

## GitHub Workflow

1. **Clone or Fork** a repository
2. **Create a branch** for your feature/fix
3. **Make changes** to files
4. **Commit** your changes with descriptive messages
5. **Push** your branch to GitHub
6. **Create a Pull Request** to propose your changes
7. **Review and Merge** after approval

## Best Practices

### Commit Messages

- Write clear, descriptive commit messages
- Use present tense: "Add feature" not "Added feature"
- Keep messages concise but informative

### Branch Naming

- Use descriptive names: `feature/user-authentication`
- Common prefixes: `feature/`, `fix/`, `bugfix/`, `hotfix/`

### .gitignore

- Create a `.gitignore` file to exclude files from version control
- Common entries: `node_modules/`, `.env`, `*.log`, `dist/`

## Useful GitHub Features

### Issues

- Track bugs, feature requests, and tasks
- Can be assigned to team members
- Can be linked to pull requests

### Actions

- Automate workflows (CI/CD)
- Run tests automatically
- Deploy code automatically

### Forking

- Create your own copy of someone else's repository
- Make changes without affecting the original
- Can submit pull requests back to the original

## Common Workflows

### Feature Development

1. Create feature branch from main
2. Develop and commit changes
3. Push branch to GitHub
4. Create pull request
5. Address review comments
6. Merge to main after approval

### Bug Fix

1. Create bugfix branch
2. Fix the issue
3. Test the fix
4. Commit and push
5. Create pull request
6. Merge after review

## Tips

- **Pull before push**: Always pull latest changes before pushing
- **Small commits**: Make frequent, small commits rather than large ones
- **Review before merge**: Always review code before merging
- **Use branches**: Don't work directly on main/master branch
- **Write good README**: Document your project clearly

## Resources

- [Picsart GitHub Playlist](https://youtube.com/playlist?list=PLBzxnGCN6T8cE3d1heX9T6XVhcudvbiYQ&si=ndaCNB56n_p-_zUB)
- [GitHub Docs](https://docs.github.com/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
