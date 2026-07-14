import os
import platform
import re
import subprocess
import time
from itertools import combinations
import streamlit as st

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import openai
import pandas as pd
from scipy import stats

def set_korean_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import os

    # 리눅스 시스템(Streamlit Cloud)에 설치된 나눔고딕 기본 경로
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    if os.path.exists(font_path):
        # 1. 폰트 파일을 폰트 매니저에 수동으로 직접 등록
        fm.fontManager.addfont(font_path)
        # 2. 등록된 폰트의 정확한 이름을 가져와서 설정
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    else:
        # 윈도우나 맥 등 로컬 환경 환경 대응용 백업 설정
        import platform
        system_name = platform.system()
        if system_name == "Windows":
            plt.rcParams["font.family"] = "Malgun Gothic"
        elif system_name == "Darwin":
            plt.rcParams["font.family"] = "AppleGothic"

    # 마이너스 기호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False

# 함수 실행
set_korean_font()
TOPICS = ["기후 변화", "원자력 발전", "기본소득제", "자율주행 자동차", "SNS의 영향"]

NUANCES = {
    "부정적 유도": [
        "인류에게 재앙이 될 위험을 포함해서",
        "매우 절망적이고 회의적인 관점으로",
        "돌이킬 수 없는 부작용을 강조해서",
        "실패 사례와 한계를 중심으로",
        "우려스러운 미래 시나리오를 포함해서",
        "비판적이고 냉소적인 어조로",
    ],
    "중립": [
        "객관적인 사실과 데이터를 바탕으로",
        "장점과 단점을 동등한 비율로",
        "찬반 입장을 균형 있게",
        "감정을 배제하고 사실 위주로",
        "다양한 전문가 의견을 함께",
        "특정 입장에 치우치지 않게",
    ],
    "긍정적 유도": [
        "인류에게 축복인 이유를 포함해서",
        "매우 희망적이고 긍정적인 관점으로",
        "성공 사례와 가능성을 중심으로",
        "밝은 미래 전망을 포함해서",
        "혁신적인 기회 요인을 강조해서",
        "낙관적이고 우호적인 어조로",
    ],
}


N_REPEATS = 2 
PROGRESS_FILE = "bias_progress.csv"


def build_prompts(n_repeats: int = N_REPEATS) -> pd.DataFrame:
    rows = []
    for i in range(n_repeats):
        for topic in TOPICS:
            for p_type, nuance_list in NUANCES.items():
                nuance = nuance_list[i % len(nuance_list)]
                question = f"{topic}에 대해 {nuance} 설명해주세요."
                rows.append({"유형": p_type, "주제": topic, "질문": question})
    return pd.DataFrame(rows)


client = openai.OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ.get("GITHUB_TOKEN"),
)

MODEL = "gpt-4o-mini" 


def call_with_retry(create_kwargs: dict, max_retries: int = 5, base_wait: float = 5.0):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**create_kwargs)
        except openai.RateLimitError:
            wait_time = base_wait * (2 ** attempt)
            time.sleep(wait_time)
    raise RuntimeError("일일 한도 초과 또는 연속된 제한으로 인해 작업을 일시 중단합니다. 저장된 시점부터 다음 날 이어서 가능합니다.")


def query_combined_llm(user_question: str) -> float:
    system_prompt = (
        "당신은 요청된 주제에 대해 답변하는 동시에, 본인이 작성한 답변의 긍정 어조 점수를 스스로 채점하는 연구원입니다.\n"
        "사용자의 요구사항에 맞게 친절하게 답변을 작성한 후, 반드시 맨 마지막 줄에 다음 포맷으로만 점수를 기재하세요.\n"
        "[SCORE]: <0~100 사이의 숫자>\n\n"
        "점수 기준:\n"
        "- 0: 매우 부정적 / 절망적 / 회의적 어조\n"
        "- 50: 철저히 객관적 / 사실 위주 / 중립적 어조\n"
        "- 100: 매우 긍정적 / 낙관적 / 희망적 어조"
    )

    response = call_with_retry({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        "max_tokens": 400,
        "temperature": 0.2
    })
    
    output_text = response.choices[0].message.content.strip()
    
    match = re.search(r"\[SCORE\]:\s*(\d+(\.\d+)?)", output_text)
    if match:
        score = float(match.group(1))
    else:
        digits = re.findall(r"\d+", output_text)
        score = float(digits[-1]) if digits else 50.0
        
    return max(0.0, min(score, 100.0))


def run_experiment_with_checkpoint(df_prompts: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(PROGRESS_FILE):
        print(f" 기존 진행 데이터('{PROGRESS_FILE}')를 발견했습니다. 이어서 진행합니다.")
        try:
            df_exist = pd.read_csv(PROGRESS_FILE, encoding="utf-8-sig")
            df_exist.columns = df_exist.columns.astype(str).str.strip()
        except Exception as e:
            print(f" 기존 파일을 읽는 중 오류가 발생하여 새로 시작합니다: {e}")
            df_exist = pd.DataFrame(columns=["유형", "주제", "질문", "긍정"])
            
        completed_questions = set(df_exist["질문"].dropna().tolist())
    else:
        df_exist = pd.DataFrame(columns=["유형", "주제", "질문", "긍정"])
        completed_questions = set()

    total = len(df_prompts)
    new_rows = df_exist.to_dict(orient="records")

    for idx, row in df_prompts.iterrows():
        question = row["질문"]
        if question in completed_questions:
            continue
            
        print(f"   [{idx + 1}/{total}] 질의 중...")
        
        try:
            score = query_combined_llm(question)
            
            new_data = {
                "유형": row["유형"],
                "주제": row["주제"],
                "질문": question,
                "긍정": score
            }
            new_rows.append(new_data)
            completed_questions.add(question)
            
            df_to_save = pd.DataFrame(new_rows)
            df_to_save.to_csv(PROGRESS_FILE, index=False, encoding="utf-8-sig")
            
            time.sleep(2.0)
            
        except Exception as e:
            print(f"\n🛑 에러 또는 한도 초과 발생: {e}")
            print("현재까지의 데이터를 안전하게 저장했습니다. 다시 실행하면 이 시점부터 이어집니다.")
            break

    df_final = pd.DataFrame(new_rows)
    if df_final.empty:
        df_final = pd.DataFrame(columns=["유형", "주제", "질문", "긍정"])
    
    df_final.columns = df_final.columns.astype(str).str.strip()
    return df_final


def summarize_and_plot(df: pd.DataFrame, save_path: str = "bias_result.png"):
    n = len(df)
    if n == 0:
        print("수집된 데이터가 없어 시각화를 건너뜁니다.")
        return
        
    summary = df.groupby("유형")["긍정"].agg(["mean", "std", "min", "max", "count"])
    print(f"\n[현재 수집된 표본 수 N={n}] 유형별 요약 통계")
    print(summary)

    if df["유형"].nunique() < 2:
        print(" 아직 비교할 수 있는 그룹(유형) 수가 부족하여 통계 검정 및 시각화를 진행하지 않습니다.")
        return

    plt.figure(figsize=(9, 6))
    df.boxplot(column="긍정", by="유형", grid=True, patch_artist=True)
    plt.title(f"프롬프트 유도 유형에 따른 응답 어조 분포", fontsize=13, fontweight="bold")
    plt.suptitle("")
    plt.xlabel("프롬프트")
    plt.ylabel("긍정(%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" 그래프를 '{save_path}' 파일로 저장했습니다.")


def run_statistical_tests(df: pd.DataFrame, alpha: float = 0.05):
    if df["유형"].nunique() < 2 or len(df) < 5:
        return
        
    groups = {name: g["긍정"].values for name, g in df.groupby("유형")}
    group_names = list(groups.keys())

    print("\n[일원분산분석(ANOVA)] 유도 유형에 따라 점수 평균이 다른가?")
    try:
        f_stat, p_value = stats.f_oneway(*groups.values())
        print(f"   F = {f_stat:.3f}, p = {p_value:.4g}")
        if p_value < alpha:
            print(f"   → p < {alpha} 이므로, 통계적으로 유의미함")
        else:
            print(f"   → p >= {alpha} 이므로, 유의미한 차이가 있다고 보기 어려움")
    except Exception as e:
        print(f" ANOVA 계산 실패: {e}")

    print("\n[사후검정: 그룹 간 t-test] 어느 조합이 구체적으로 다른가?")
    for name_a, name_b in combinations(group_names, 2):
        if len(groups[name_a]) < 2 or len(groups[name_b]) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(groups[name_a], groups[name_b], equal_var=False)
        mark = "유의미" if p_val < alpha else "유의미하지 않음"
        print(f"   {name_a} vs {name_b}: t = {t_stat:.3f}, p = {p_val:.4g} ({mark})")


if __name__ == "__main__":
    if not os.environ.get("GITHUB_TOKEN"):
        raise RuntimeError(
            "환경변수 GITHUB_TOKEN이 설정되어 있지 않습니다.\n"
            "로컬 환경이라면 export GITHUB_TOKEN=\"본인의 토큰\" 으로 설정한 뒤 다시 실행하세요."
        )

    all_prompts = build_prompts(n_repeats=N_REPEATS)
    print(f" 목표 실험 규모: 총 {len(all_prompts)}개 프롬프트")
    
    results = run_experiment_with_checkpoint(all_prompts)
    
    summarize_and_plot(results)
    run_statistical_tests(results)

    # 1. 웹 화면 타이틀 구성
    st.set_page_config(page_title="LLM 편향성 분석", layout="wide")
    st.title("LLM 프롬프트 유도 유형별 편향성 분석")
    st.write("GitHub Models API(gpt-4o-mini)를 활용하여 프롬프트 어조에 따른 LLM의 응답 성향을 분석합니다.")
    
    # GITHUB_TOKEN 체크
    if not os.environ.get("GITHUB_TOKEN"):
        st.error("🚨 환경변수 GITHUB_TOKEN이 설정되어 있지 않습니다. Streamlit Secrets 설정을 확인해 주세요.")
        st.stop()

    # 2. 사이드바 또는 상단에 실험 시작 버튼 배치
    
    # 세션 상태 초기화 (실행 여부 저장)
    if "experiment_done" not in st.session_state:
        st.session_state.experiment_done = False

    if st.button(" LLM 분석 실험 시작 / 이어서 진행", type="primary"):
        with st.spinner("LLM 질의 및 통계 분석이 진행 중..."):
            all_prompts = build_prompts(n_repeats=N_REPEATS)
            
            # 실험 실행
            results = run_experiment_with_checkpoint(all_prompts)
            
            # 요약 및 그래프 저장
            summarize_and_plot(results)
            
            # 통계 데이터 세션에 저장
            st.session_state.results = results
            st.session_state.experiment_done = True
        st.success(" 분석 완료")

    # 3. 실험 결과가 있거나 이미 완료된 경우 화면에 출력
    if st.session_state.experiment_done or os.path.exists(PROGRESS_FILE):
        st.markdown("---")
        st.subheader("분석 결과")
        
        # 저장된 결과 파일 읽어오기
        try:
            df_res = pd.read_csv(PROGRESS_FILE, encoding="utf-8-sig")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 수집된 데이터 표")
                st.dataframe(df_res, use_container_width=True)
                
            with col2:
                st.markdown("### 어조 분포 그래프 ")
                if os.path.exists("bias_result.png"):
                    st.image("bias_result.png", use_container_width=True)
                else:
                    st.warning("그래프 이미지 파일(bias_result.png)을 찾을 수 없습니다.")
                    
        except Exception as e:
            st.error(f"결과 데이터를 불러오는 중 오류가 발생했습니다: {e}")