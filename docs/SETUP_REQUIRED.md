# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## GitHub Workflow Setup

⚠️ **CRITICAL**: The GitHub App lacks permissions to create workflow files. Repository maintainers must manually create these files.

### Step 1: Create Workflow Directory

Create the following directory structure in your repository:

```
.github/
└── workflows/
```

### Step 2: Copy Workflow Files

Copy the following files from `docs/workflows/examples/` to `.github/workflows/`:

1. **ci.yml** - Main CI pipeline
2. **security-scan.yml** - Security scanning and compliance
3. **cd.yml** - Continuous deployment (when created)
4. **dependency-update.yml** - Automated dependency updates (when created)
5. **benchmarks.yml** - Performance benchmarking (when created)

### Step 3: Configure Repository Secrets

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Container Registry
```
DOCKER_HUB_USERNAME=your_docker_username
DOCKER_HUB_TOKEN=your_docker_token
```

#### Cloud Deployment
```
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
GCP_SA_KEY=your_gcp_service_account_key
```

#### Monitoring and Analytics
```
WANDB_API_KEY=your_wandb_key
SLACK_WEBHOOK_URL=your_slack_webhook
```

#### Security Scanning
```
SNYK_TOKEN=your_snyk_token
SONAR_TOKEN=your_sonar_token
```

#### Package Registry
```
PYPI_API_TOKEN=your_pypi_token
```

### Step 4: Configure Branch Protection

Configure branch protection rules for `main` branch:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require a pull request before merging
   - Require approvals (2 recommended)
   - Dismiss stale PR approvals when new commits are pushed
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Required status checks:
     - `CI Success`
     - `Security Summary`
     - `Code Quality`

### Step 5: Setup Self-Hosted GPU Runners (Optional)

For GPU testing, set up self-hosted runners:

1. Go to Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Follow setup instructions for Linux
4. Label the runner with: `gpu`, `linux`, `cuda`

## Repository Settings

### Actions Permissions

1. Go to Settings → Actions → General
2. Set "Actions permissions" to:
   - "Allow all actions and reusable workflows"
3. Set "Workflow permissions" to:
   - "Read and write permissions"
   - "Allow GitHub Actions to create and approve pull requests"

### Environment Setup

Create deployment environments:

1. Go to Settings → Environments
2. Create environments:
   - **staging**: Require reviewers (DevOps team)
   - **production**: Require reviewers (Admin team), 60-minute delay

## Additional Configurations

### Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with:
- bug_report.yml
- feature_request.yml
- security_report.yml

### Pull Request Template

Create `.github/pull_request_template.md`

### CODEOWNERS

Create `.github/CODEOWNERS` file to auto-assign reviewers.

## Verification Checklist

After manual setup, verify:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] Actions permissions are set correctly
- [ ] Environment protection rules are configured
- [ ] GPU runners are connected (if applicable)

## Support

If you need assistance with the manual setup:

1. Check the documentation in `docs/workflows/README.md`
2. Review example configurations in `docs/workflows/examples/`
3. Contact the development team
4. File an issue with the `setup-help` label

## Next Steps

Once manual setup is complete:

1. Test the CI pipeline with a test PR
2. Verify all status checks pass
3. Confirm security scans execute properly
4. Test deployment to staging environment
5. Document any additional customizations needed

---

**Note**: This manual setup is a one-time requirement. Once configured, the SDLC automation will function normally.