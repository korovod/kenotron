# How to contribute to ⚡️ Kénotron?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](CODE_OF_CONDUCT.md).

## You can contribute in so many ways!

Some of the ways you can contribute to Kénotron:

* Fixing outstanding issues with the existing code;
* Contributing to the examples or to the documentation;
* Submitting issues related to bugs or desired new features.

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The Kénotron library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

Did not find it? :( So we can act quickly on it, please follow these steps:

* Include your **OS type and version**, the versions of **Python** and **PyTorch**.
* A short, self-contained, code snippet that allows us to reproduce the bug in a few minutes;
* Provide your Kénotron configuration used for the run;
* Describe the 1) expected behavior and 2) the actual behavior;

### Do you want a new feature?

A good feature request addresses the following points:

1. Motivation first:
* Is it related to a problem/frustration with the library? If so, please explain
  why. Providing a code snippet that demonstrates the problem is best.
* Is it related to something you would need for a project? We'd love to hear
  about it!
* Is it something you worked on and think could benefit the community?
  Awesome! Tell us what problem it solved for you.

2. Write a *full paragraph* describing the feature;

3. Provide a **code snippet** that demonstrates its future use;

4. If your feature is related to the implementation of a parallelism technique, make sure to read our [converging guidelines](./docs/converging.md);

5. In case this is related to a paper, please attach a link;

6. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Submitting a pull request (PR)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
Kénotron. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/korovod/kenotron) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote. The following command
   assumes you have your public SSH key uploaded to GitHub. See the following guide for more
   [information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

   ```bash
   git clone git@github.com:<your Github handle>/kenotron.git
   cd kenotron
   git remote add upstream https://github.com/korovod/kenotron.git
   ```

3. Create a new branch to hold your development changes, and do this for every new PR you work on.

   Start by synchronizing your `main` branch with the `upstream/main` branch (more details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a conda or a virtual environment you've created for working on this library:

   ```bash
   uv venv
   uv pip install -e ".[dev]"
   uv pip install -e ".[test]"
   pre-commit install
   ```

   (If Kénotron was already installed in the virtual environment, remove
   it with `uv pip uninstall nanotron` before reinstalling it in editable
   mode with the `-e` flag.)

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this (see
   below an explanation regarding the environment variable):

   ```bash
   python -m pytest tests/<TEST_TO_RUN>.py
   ```

   This target is also optimized to only work with files modified by the PR you're working on.

   If you prefer to run the checks one after the other, the following command apply the
   style corrections:

   ```bash
   pre-commit run --all-files
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please write a commit message using the [Conventional Commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/). Ideally, a longer commit body is provided after the short description, providing additional contextual information about the code changes.

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`, or mark
   the PR as a draft PR. These are useful to avoid duplicated work, and to differentiate
   it from PRs ready to be merged;
4. If your work is related to the implementation of a parallelism technique, please report
   some results to make sure that convergence is preserved, as explained in the
   [converging guidelines](./docs/converging.md).
5. Make sure existing tests pass;
6. Add high-coverage tests. No quality testing = no merge.

See an example of a good PR here: https://github.com/huggingface/nanotron/pull/155

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests folder](https://github.com/korovod/nanotron/tree/main/tests).

We use `pytest` in order to run the tests. From the root of the
repository, here's how to run tests with `pytest` for the library:

```bash
# Runs all tests (where 12 of which run in parallel)
$ pytest -n 12 tests
```

You can specify a smaller set of tests in order to test only the feature you're working on.
