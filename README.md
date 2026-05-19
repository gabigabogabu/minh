# minh

MINimial Harness is a very small llm harness. Based on bun, ai sdk and openrouter.

## Usage

Skip Permissions

`% yes | ./minh.ts "What's in this dir?"`

List Models

`% ./minh.ts --list-models | grep openai`

```sh
% ./minh.ts "What's in this dir?"
Using chat file: ./1779141354787.chat
Using model: openrouter/free

<thinking>The user is asking what's in the current directory. I should list the files and directories in the current working directory to see what's there.
<tool> $ find . -maxdepth 2 -type f | head -50 </tool></thinking>
Can I run `find . -maxdepth 2 -type f | head -50` (y/N)? y

<tool-result>./1779138259504.chat
./minh.ts
./bun.lock
./README.md
./.gitignore
./package.json
./.env
./tsconfig.json
./.git/REBASE_HEAD
./.git/ORIG_HEAD
./.git/config
./.git/HEAD
./.git/description
./.git/index
./.git/COMMIT_EDITMSG
./.git/FETCH_HEAD
./1779141354787.chat
</tool-result>

<thinking>The user asked "What's in this dir?" which is a simple request to list the contents of the current directory. I used the `find` command to list files up to a depth of 2, but the output shows a reasonable set of files and directories. Let me provide a clear summary of what's in the directory.</thinking>

Here's what's in the current directory:

**Files:**
- `1779138259504.chat`
- `minh.ts`
- `bun.lock`
- `README.md`
- `.gitignore`
- `package.json`
- `.env`
- `tsconfig.json`
- `.git/REBASE_HEAD`
- `.git/ORIG_HEAD`
- `.git/config`
- `.git/HEAD`
- `.git/description`
- `.git/index`
- `.git/COMMIT_EDITMSG`
- `.git/FETCH_HEAD`
- `1779141354787.chat`

This appears to be a TypeScript project with a `.git` repository, package configuration, and some chat files.
Usage: 957 tokens
  input:  698 (698 new, 0 cached read, 0 cached write)
  output: 259 (163 text, 96 reasoning)
```
