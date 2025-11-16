#!/bin/bash
# Configure Route 53 DNS for custom domain

set -e

# Load configuration
source aws-infrastructure/config.env

echo "🌐 Configuring DNS for $DOMAIN_NAME..."

# Get hosted zone ID for opkclodz.com
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
    --dns-name opkcloudz.com \
    --query "HostedZones[0].Id" \
    --output text | cut -d'/' -f3)

if [ -z "$HOSTED_ZONE_ID" ] || [ "$HOSTED_ZONE_ID" = "None" ]; then
    echo "❌ Hosted zone for opkcloudz.com not found!"
    echo "Please create a hosted zone first:"
    echo "  aws route53 create-hosted-zone --name opkcloudz.com --caller-reference $(date +%s)"
    exit 1
fi

echo "Hosted Zone ID: $HOSTED_ZONE_ID"

# Get ALB Hosted Zone ID
ALB_HOSTED_ZONE=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query "LoadBalancers[0].CanonicalHostedZoneId" \
    --output text \
    --region $AWS_REGION)

# Create/Update DNS record
cat > /tmp/dns-change.json << DNS
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
          "EvaluateTargetHealth": true
        }
      }
    }
  ]
}
DNS

# Apply DNS change
CHANGE_ID=$(aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch file:///tmp/dns-change.json \
    --query 'ChangeInfo.Id' \
    --output text)

echo "✅ DNS record created/updated"
echo "Change ID: $CHANGE_ID"
echo ""
echo "⏳ Waiting for DNS propagation (this may take 1-2 minutes)..."

aws route53 wait resource-record-sets-changed --id $CHANGE_ID

echo "✅ DNS propagation complete!"
echo ""
echo "🎉 Your API is now accessible at:"
echo "  http://$DOMAIN_NAME"
echo ""
echo "Test with:"
echo "  curl http://$DOMAIN_NAME/health"
echo ""
