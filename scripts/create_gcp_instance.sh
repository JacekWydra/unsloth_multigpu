#!/bin/bash
# Create GCP instances for multi-GPU testing in Warsaw region (europe-central2)
set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
ZONES=("europe-west1-b" "europe-west1-c" "europe-west1-d" "us-central1-a" "us-central1-b" "us-central1-c")  # Available T4 zones
INSTANCE_NAME_PREFIX="unsloth-multigpu-test"
MACHINE_TYPE_2GPU="n1-standard-8"
MACHINE_TYPE_4GPU="n1-standard-16"
# Alternative machine types if primary ones fail
ALT_MACHINE_TYPE_2GPU=("n1-highmem-8" "n2-standard-8" "n2-highmem-8" "c2-standard-16")
ALT_MACHINE_TYPE_4GPU=("n1-highmem-16" "n2-standard-16" "n2-highmem-16" "c2-standard-30")
GPU_TYPE="nvidia-tesla-t4"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"
DISK_TYPE="pd-ssd"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <project-id> [2gpu|4gpu|both] [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 my-project 2gpu          # Create 2-GPU instance"
    echo "  $0 my-project 4gpu          # Create 4-GPU instance" 
    echo "  $0 my-project both          # Create both instances"
    echo "  $0 my-project both --dry-run # Show commands without executing"
    echo ""
    echo "Region: europe-west1 (Belgium) and us-central1 (Iowa) - T4 available zones"
    echo "GPU Type: nvidia-tesla-t4"
}

create_instance() {
    local gpu_count=$1
    local instance_name="${INSTANCE_NAME_PREFIX}-${gpu_count}gpu"
    local machine_types
    
    if [ "$gpu_count" = "2" ]; then
        machine_types=("$MACHINE_TYPE_2GPU" "${ALT_MACHINE_TYPE_2GPU[@]}")
    elif [ "$gpu_count" = "4" ]; then
        machine_types=("$MACHINE_TYPE_4GPU" "${ALT_MACHINE_TYPE_4GPU[@]}")
    else
        echo -e "${RED}Invalid GPU count: $gpu_count${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Creating $gpu_count-GPU instance: $instance_name${NC}"
    
    # Try each zone and machine type combination
    for zone in "${ZONES[@]}"; do
        for machine_type in "${machine_types[@]}"; do
            echo -e "${YELLOW}Trying zone: $zone, machine type: $machine_type${NC}"
            
            local cmd="gcloud compute instances create $instance_name \\
            --project=$PROJECT_ID \\
            --zone=$zone \\
            --machine-type=$machine_type \\
            --accelerator=type=$GPU_TYPE,count=$gpu_count \\
            --maintenance-policy=TERMINATE \\
            --provisioning-model=STANDARD \\
            --service-account=default \\
            --scopes=https://www.googleapis.com/auth/cloud-platform \\
            --image-family=$IMAGE_FAMILY \\
            --image-project=$IMAGE_PROJECT \\
            --boot-disk-size=$BOOT_DISK_SIZE \\
            --boot-disk-type=$DISK_TYPE \\
            --boot-disk-device-name=$instance_name \\
            --reservation-affinity=any \\
            --metadata=install-nvidia-driver=True \\
            --tags=gpu-instance,unsloth-test"
            
            if [ "$DRY_RUN" = "true" ]; then
                echo "DRY RUN - Would execute:"
                echo "$cmd"
                echo ""
                return 0  # Don't try all combinations in dry run
            else
                if eval $cmd 2>/dev/null; then
                    echo -e "${GREEN}✓ Successfully created $instance_name in $zone with $machine_type${NC}"
                    return 0
                else
                    echo -e "${YELLOW}Failed in zone $zone with $machine_type, trying next option...${NC}"
                fi
            fi
        done
    done
    
    echo -e "${RED}✗ Failed to create $instance_name in any zone/machine type combination${NC}"
    return 1
}

wait_for_ssh() {
    local instance_name=$1
    echo -e "${YELLOW}Waiting for SSH access to $instance_name...${NC}"
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if gcloud compute ssh $instance_name --zone=$ZONE --project=$PROJECT_ID --command="echo 'SSH ready'" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ SSH access ready for $instance_name${NC}"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting..."
        sleep 10
        ((attempt++))
    done
    
    echo -e "${RED}✗ SSH access timeout for $instance_name${NC}"
    return 1
}

setup_instance() {
    local instance_name=$1
    echo -e "${YELLOW}Setting up $instance_name...${NC}"
    
    # Copy setup script to instance
    gcloud compute scp setup_instance.sh $instance_name:~ --zone=$ZONE --project=$PROJECT_ID
    
    # Execute setup script
    gcloud compute ssh $instance_name --zone=$ZONE --project=$PROJECT_ID --command="chmod +x setup_instance.sh && ./setup_instance.sh"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Setup completed for $instance_name${NC}"
    else
        echo -e "${RED}✗ Setup failed for $instance_name${NC}"
        return 1
    fi
}

# Parse arguments
if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

PROJECT_ID=$1
GPU_CONFIG=$2
DRY_RUN=false

if [ "$3" = "--dry-run" ]; then
    DRY_RUN=true
fi

# Validate GPU configuration
if [ "$GPU_CONFIG" != "2gpu" ] && [ "$GPU_CONFIG" != "4gpu" ] && [ "$GPU_CONFIG" != "both" ]; then
    echo -e "${RED}Invalid GPU configuration: $GPU_CONFIG${NC}"
    print_usage
    exit 1
fi

echo -e "${GREEN}=== GCP Multi-GPU Instance Creation ===${NC}"
echo "Project ID: $PROJECT_ID"
echo "Zone: $ZONE"
echo "GPU Configuration: $GPU_CONFIG"
echo "Dry Run: $DRY_RUN"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
    echo -e "${RED}Error: gcloud is not authenticated. Run 'gcloud auth login' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ gcloud authenticated${NC}"

# Set project
echo -e "${YELLOW}Using project: $PROJECT_ID${NC}"
echo -e "${GREEN}✓ Project configured${NC}"

# Check quotas
echo -e "${YELLOW}Checking GPU quotas...${NC}"
if [ "$DRY_RUN" != "true" ]; then
    echo "Retrieving project quotas..."
    QUOTA_INFO=$(gcloud compute project-info describe --project=$PROJECT_ID --format="get(quotas)" 2>/dev/null)
    
    if echo "$QUOTA_INFO" | grep -q "GPUS_ALL_REGIONS"; then
        GPU_SECTION=$(echo "$QUOTA_INFO" | grep -o "[^;]*GPUS_ALL_REGIONS[^;]*")
        GPU_LIMIT=$(echo "$GPU_SECTION" | sed -n "s/.*'limit': \([0-9.]*\).*/\1/p")
        GPU_USAGE=$(echo "$GPU_SECTION" | sed -n "s/.*'usage': \([0-9.]*\).*/\1/p")
        
        if [ -n "$GPU_LIMIT" ]; then
            echo "GPU Quota - Limit: ${GPU_LIMIT} GPUs, Used: ${GPU_USAGE:-0} GPUs"
            
            # Simple validation for requested configuration
            GPU_LIMIT_INT=${GPU_LIMIT%.*}  # Remove decimal part
            REQUESTED_GPUS=0
            case "$GPU_CONFIG" in
                "2gpu") REQUESTED_GPUS=2 ;;
                "4gpu") REQUESTED_GPUS=4 ;;
                "both") REQUESTED_GPUS=6 ;;  # 2 + 4 GPUs total
            esac
            
            if [ "$GPU_LIMIT_INT" -ge "$REQUESTED_GPUS" ]; then
                echo -e "${GREEN}✓ Sufficient quota for $GPU_CONFIG configuration${NC}"
            else
                echo -e "${YELLOW}⚠ Warning: Requested $REQUESTED_GPUS GPUs but quota limit is $GPU_LIMIT${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ Could not parse GPU quota information${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No GPUS_ALL_REGIONS quota found${NC}"
        echo "Note: You may need to request GPU quota increases in the Google Cloud Console"
    fi
else
    echo "DRY RUN - Skipping quota check"
fi

# Create instances
echo -e "${YELLOW}Starting instance creation...${NC}"

if [ "$GPU_CONFIG" = "2gpu" ] || [ "$GPU_CONFIG" = "both" ]; then
    echo -e "${YELLOW}Creating 2-GPU instance...${NC}"
    create_instance 2
fi

if [ "$GPU_CONFIG" = "4gpu" ] || [ "$GPU_CONFIG" = "both" ]; then
    echo -e "${YELLOW}Creating 4-GPU instance...${NC}"
    create_instance 4
fi

if [ "$DRY_RUN" != "true" ]; then
    echo -e "${GREEN}=== Instance Creation Complete ===${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Wait for instances to boot (2-3 minutes)"
    echo "2. Run setup script: ./setup_instance.sh"
    echo "3. SSH to instances and run tests"
    echo ""
    echo "SSH commands:"
    
    if [ "$GPU_CONFIG" = "2gpu" ] || [ "$GPU_CONFIG" = "both" ]; then
        echo "  gcloud compute ssh ${INSTANCE_NAME_PREFIX}-2gpu --zone=$ZONE --project=$PROJECT_ID"
    fi
    
    if [ "$GPU_CONFIG" = "4gpu" ] || [ "$GPU_CONFIG" = "both" ]; then
        echo "  gcloud compute ssh ${INSTANCE_NAME_PREFIX}-4gpu --zone=$ZONE --project=$PROJECT_ID"
    fi
fi