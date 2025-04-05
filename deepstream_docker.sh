

docker run --gpus all -d \
    -v /home/godsublab/.ssh:/root/.ssh \
    nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
    deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt

# git ssh set up
# SSH 키 권한 설정
chmod 600 /root/.ssh/id_ed25519
chmod 644 /root/.ssh/id_ed25519.pub

# SSH 에이전트 시작 및 키 추가
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_ed25519

# GitHub SSH 연결 테스트
ssh -T git@github.com

# 사용자 및 이메일 계정 설정
git config --global user.name "mskim46"
git config --global user.email "mskim@rgblab.kr"

# 레포지토리 클론
git clone git@github.com:mskim46/DeepStream-ROI-Template-Tracking.git
