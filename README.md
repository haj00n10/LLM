# LLM 프롬프트 유도에 따른 어조 편향성 검정 실험

프롬프트의 지시사항(긍정, 중립, 부정 유도)에 따라 LLM(gpt-4o-mini)이 생성하는 답변의 어조가 실제로 편향되는지 정량적으로 수집하고 통계적으로 가설을 검정하는 스크립트


## 시스템 요구사항
 `openai`, `pandas`, `matplotlib`, `scipy`,고딕체 설치

```bash
pip install openai pandas matplotlib scipy
sudo apt-get install -y fonts-nanum-extra