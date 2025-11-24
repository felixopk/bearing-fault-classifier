#!/bin/bash
# Configure HTTPS + ACM + DNS for API domain

set -euo pipefail

source aws-infrastructure/config.env

echo "🌍 Setting up HTTPS + Route53 DNS for: $DOMAIN_NAME"
echo ""

# -----------------------
# 1. REQUEST ACM CERTIFICATE
# -----------------------
echo "📜 Requesting ACM Certificate..."

CERT_ARN=$(aws acm request-certificate \
    --domain-name "$DOMAIN_NAME" \
    --validation-method DNS \
    --query CertificateArn \
    --output text \
    --region "$AWS_REGION")

echo "  Certificate ARN: $CERT_ARN"
echo ""

# -----------------------
# 2. GET DNS VALIDATION RECORDS
# -----------------------
echo "🔍 Getting DNS validation details..."

DNS_RECORD=$(aws acm describe-certificate \
    --certificate-arn "$CERT_ARN" \
    --query "Certificate.DomainValidationOptions[0].ResourceRecord" \
    --output json \
    --region "$AWS_REGION")

VALIDATION_NAME=$(echo "$DNS_RECORD" | jq -r '.Name')
VALIDATION_VALUE=$(echo "$DNS_RECORD" | jq -r '.Value')

echo "  Validation Name:  $VALIDATION_NAME"
echo "  Validation Value: $VALIDATION_VALUE"
echo ""

# -----------------------
# 3. FIND HOSTED ZONE ID
# -----------------------
BASE_DOMAIN=$(echo "$DOMAIN_NAME" | sed 's/^[^.]*\.//')

HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
    --dns-name "$BASE_DOMAIN" \
    --query "HostedZones[0].Id" \
    --output text | cut -d'/' -f3)

if [[ "$HOSTED_ZONE_ID" == "" || "$HOSTED_ZONE_ID" == "None" ]]; then
  echo "❌ Hosted zone for $BASE_DOMAIN not found!"
  exit 1
fi

echo "  Hosted Zone ID: $HOSTED_ZONE_ID"
echo ""

# -----------------------
# 4. CREATE DNS VALIDATION RECORD
# -----------------------
echo "📝 Creating DNS validation record for ACM..."

cat > /tmp/acm-validation.json << EOF
{
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "$VALIDATION_NAME",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [
          { "Value": "$VALIDATION_VALUE" }
        ]
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --change-batch file:///tmp/acm-validation.json

echo "  DNS validation record created!"
echo ""

# -----------------------
# 5. WAIT FOR CERT VALIDATION
# -----------------------
echo "⏳ Waiting for ACM to validate certificate (2–5 minutes)..."

aws acm wait certificate-validated \
    --certificate-arn "$CERT_ARN" \
    --region "$AWS_REGION"

echo "  🎉 ACM certificate validated!"
echo ""

# -----------------------
# 6. CREATE HTTPS LISTENER (443)
# -----------------------
echo "⚖️ Creating HTTPS listener on ALB..."

aws elbv2 create-listener \
  --load-balancer-arn "$ALB_ARN" \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn="$CERT_ARN" \
  --ssl-policy ELBSecurityPolicy-2016-08 \
  --default-actions Type=forward,TargetGroupArn="$TG_ARN" \
  --region "$AWS_REGION" 2>/dev/null || echo "✔️ HTTPS listener already exists"

echo ""

# -----------------------
# 7. CREATE A RECORD FOR API → ALB
# -----------------------
echo "🌐 Creating A record for domain → ALB..."

# get ALB hosted zone
ALB_HOSTED_ZONE=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns "$ALB_ARN" \
    --query "LoadBalancers[0].CanonicalHostedZoneId" \
    --output text)

cat > /tmp/api-alias.json << EOF
{
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "$DOMAIN_NAME",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "$ALB_HOSTED_ZONE",
          "DNSName": "$ALB_DNS",
          "EvaluateTargetHealth": false
        }
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
  --hosted-zone-id "$HOSTED_ZONE_ID" \
  --change-batch file:///tmp/api-alias.json

echo "✔️ A record created: $DOMAIN_NAME → $ALB_DNS"
echo ""
echo "🎉 Your API will be available at: https://$DOMAIN_NAME"
