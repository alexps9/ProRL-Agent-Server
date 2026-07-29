# Layer a coding-agent runtime onto a Terminal-Bench / Harbor task image.
#
# BASE_IMAGE is the per-task image built straight from the task's
# environment/Dockerfile (see build_images.py). We add Node.js (the
# claude_code / codex / qwen_code / opencode CLIs are Node packages, installed
# at task time during Polar's INIT stage) and git. We stay as the base image's
# default user (root) and set no WORKDIR, so the agent sees the same container
# layout the Harbor verifier expects.
ARG BASE_IMAGE
ARG POLAR_TB_IMAGE_VERSION=1
FROM ${BASE_IMAGE}

ARG POLAR_TB_IMAGE_VERSION=1
LABEL io.polar.terminal-bench-image-version=${POLAR_TB_IMAGE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    NPM_CONFIG_UPDATE_NOTIFIER=false

# Install Node 22 UNCONDITIONALLY. Many Harbor task images apt-install an old
# distro nodejs; modern harness CLIs need Node 22+.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg git \
    && apt-get purge -y nodejs npm libnode-dev libnode72 >/dev/null 2>&1 || true \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && node --version | grep -q '^v22\.' \
    && mkdir -p /polar/session
