# PulsePoint – AWS Fargate Deployment Guide

## Overview

```
[Trained model artifacts] → S3 (model_artifacts/)
         ↓
[Docker image] → ECR
         ↓
ECS Fargate Task (pulls from ECR + S3) → runs Streamlit on port 8501
         ↓
Application Load Balancer → public URL
```

---

## Prerequisites

- AWS CLI configured (`aws configure`)
- Docker installed locally
- ECR repository created (or use the command below)
- IAM roles in place (see Step 1)

---

## Step 1 – IAM Roles

### Task Execution Role
Use the AWS-managed policy `AmazonECSTaskExecutionRolePolicy`.  
Already exists in most accounts as `ecsTaskExecutionRole`.

### Task Role (grants the container access to S3)
Create a custom role `pulsepoint-fargate-task-role` with this inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::pulsepoint-raw-zone-mena",
        "arn:aws:s3:::pulsepoint-raw-zone-mena/model_artifacts/*"
      ]
    }
  ]
}
```

---

## Step 2 – Upload Model Artifacts to S3

After running the Phase 2 notebook, upload the `model_artifacts/` folder:

```bash
aws s3 cp model_artifacts/ s3://pulsepoint-raw-zone-mena/model_artifacts/ --recursive
```

Verify:
```bash
aws s3 ls s3://pulsepoint-raw-zone-mena/model_artifacts/
```

Expected files:
- `scaler.pkl`
- `pca.pkl`
- `logistic_model.pkl`
- `model_metadata.json`

---

## Step 3 – Build & Push Docker Image to ECR

```bash
# Set your account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO=pulsepoint-streamlit

# Create ECR repo (first time only)
aws ecr create-repository --repository-name $REPO --region $REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build image
docker build -t $REPO .

# Tag & push
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

---

## Step 4 – Register ECS Task Definition

```bash
# Replace YOUR_ACCOUNT_ID in ecs_task_definition.json first, then:
aws ecs register-task-definition \
  --cli-input-json file://ecs_task_definition.json \
  --region us-east-1
```

---

## Step 5 – Create ECS Cluster & Service

```bash
# Create cluster (Fargate — no EC2 to manage)
aws ecs create-cluster --cluster-name pulsepoint-cluster --region us-east-1

# Create CloudWatch log group for container logs
aws logs create-log-group --log-group-name /ecs/pulsepoint-streamlit --region us-east-1

# Create service (replace subnet/sg IDs with yours)
aws ecs create-service \
  --cluster pulsepoint-cluster \
  --service-name pulsepoint-streamlit \
  --task-definition pulsepoint-streamlit \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
      subnets=[subnet-XXXXXXXX],
      securityGroups=[sg-XXXXXXXX],
      assignPublicIp=ENABLED
  }" \
  --region us-east-1
```

> **Security Group rule needed:** Inbound TCP 8501 from 0.0.0.0/0 (or your IP only for testing)

---

## Step 6 – Access the App

```bash
# Get the public IP of the running task
TASK_ARN=$(aws ecs list-tasks --cluster pulsepoint-cluster \
  --service-name pulsepoint-streamlit --query 'taskArns[0]' --output text)

ENI=$(aws ecs describe-tasks --cluster pulsepoint-cluster --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo "App URL: http://$PUBLIC_IP:8501"
```

---

## Updating the App (Re-deploy)

No model re-training needed. To push a UI change:

```bash
docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest

# Force new task deployment
aws ecs update-service \
  --cluster pulsepoint-cluster \
  --service pulsepoint-streamlit \
  --force-new-deployment \
  --region us-east-1
```

---

## Cost Estimate (AWS Learner Lab)

| Resource | Spec | Est. Cost |
|---|---|---|
| Fargate task | 0.5 vCPU, 1 GB RAM | ~$0.015/hr |
| ECR storage | ~500 MB image | ~$0.05/mo |
| S3 model artifacts | ~5 MB | negligible |

**Stop the service when not in use:**
```bash
aws ecs update-service --cluster pulsepoint-cluster \
  --service pulsepoint-streamlit --desired-count 0
```
