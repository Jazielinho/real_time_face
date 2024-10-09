import streamlit as st
import cv2
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from models.fast_statistics import FastStatistics
import queue
import warnings
import av

warnings.filterwarnings("ignore")

st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .metric {
        font-size: 1.3em;
        font-weight: bold;
    }
    .metric-title {
        font-size: 1.1em;
        color: #bfbfbf;
    }
    .stop-btn {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stop-btn:hover {
        background-color: #ff3333;
    }
    </style>
""", unsafe_allow_html=True)

if 'fast_statistics' not in st.session_state:
    print("Creating fast statistics")
    st.session_state['fast_statistics'] = FastStatistics(use_cv2=False)
fast_statistics = st.session_state['fast_statistics']

if 'seconds' not in st.session_state:
    st.session_state['seconds'] = 0
seconds = st.session_state['seconds']

total_faces_placeholder = st.empty()

st.title("Estadísticas en tiempo real")
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    happy_placeholder = st.empty()

with col2:
    sad_placeholder = st.empty()

with col3:
    angry_placeholder = st.empty()

with col4:
    surprised_placeholder = st.empty()

with col5:
    disgust_placeholder = st.empty()

with col6:
    fear_placeholder = st.empty()

with col7:
    neutral_placeholder = st.empty()

gaze_placeholder = st.empty()

fast_statistics_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def process(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            real_time_statistics = fast_statistics.update(img)
            fast_statistics_queue.put(fast_statistics)
            for _real_time_statistic in real_time_statistics:
                x, y, w, h = _real_time_statistic['bbox']
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"Error en el procesamiento de video: {e}")
            return frame

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

def plot_last_frame(data):
    SENTIMENT_COLUMNS = [
        'emotion_angry', 'emotion_disgust', 'emotion_fear', 'emotion_happy', 'emotion_sad', 'emotion_surprise',
        'emotion_neutral'
    ]

    last_frame_id = data['frame_id'].max()
    before_frame_id = last_frame_id - 1
    last_people = len(data[data['frame_id'] == last_frame_id])
    before_people = len(data[data['frame_id'] == before_frame_id])
    total_faces_placeholder.metric(
        "Total de personas reconocidas",
        value=last_people,
        delta=last_people - before_people
    )

    last_sentiments = data[data['frame_id'] == last_frame_id][SENTIMENT_COLUMNS].mean().to_dict()
    happy_placeholder.metric("😊 Felicidad", value=f"{last_sentiments['emotion_happy']:.2f}")
    sad_placeholder.metric("😢 Tristeza", value=f"{last_sentiments['emotion_sad']:.2f}")
    angry_placeholder.metric("😠 Enojo", value=f"{last_sentiments['emotion_angry']:.2f}")
    surprised_placeholder.metric("😲 Sorpresa", value=f"{last_sentiments['emotion_surprise']:.2f}")
    disgust_placeholder.metric("🤢 Disgusto", value=f"{last_sentiments['emotion_disgust']:.2f}")
    fear_placeholder.metric("😨 Miedo", value=f"{last_sentiments['emotion_fear']:.2f}")
    neutral_placeholder.metric("😐 Neutral", value=f"{last_sentiments['emotion_neutral']:.2f}")

    last_gaze = data[data['frame_id'] == last_frame_id]['is_looking'].mean()
    gaze_placeholder.metric("👀 Mirada al centro", value=f"{last_gaze:.2f}")

def customize_plot_layout(fig):
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=12, color="white"),
        title_font=dict(size=16, color="white"),
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

@st.fragment(run_every=5)
def plot_statistics():
    total_people_chart_placeholder = st.empty()
    emotion_chart_placeholder = st.empty()
    gaze_chart_placeholder = st.empty()

    try:
        fast_statistics = fast_statistics_queue.get_nowait()
        data = fast_statistics.get_statistics()

        if data is not None:
            plot_last_frame(data)

            df_people = data[['second', 'frame_id']].groupby(['second', 'frame_id']).size().reset_index(name='count')
            df_people = df_people.groupby('second')['count'].mean().reset_index()
            fig_people = px.line(df_people, x='second', y='count', title='Personas en el tiempo')
            customize_plot_layout(fig_people)
            total_people_chart_placeholder.plotly_chart(fig_people, use_container_width=True)

            emotion_cols = [col for col in data.columns if 'emotion_' in col]
            if emotion_cols:
                df_emotions = data[['second'] + emotion_cols].groupby('second').mean().reset_index()
                fig_emotions = px.bar(df_emotions, x='second', y=emotion_cols, title='Emociones en el tiempo')
                customize_plot_layout(fig_emotions)
                emotion_chart_placeholder.plotly_chart(fig_emotions, use_container_width=True)

            if 'is_looking' in data.columns:
                df_gaze = data[['second', 'is_looking']].groupby('second').mean().reset_index()
                fig_gaze = px.line(df_gaze, x='second', y='is_looking', title='Mirada al centro en el tiempo')
                customize_plot_layout(fig_gaze)
                gaze_chart_placeholder.plotly_chart(fig_gaze, use_container_width=True)
    except queue.Empty:
        pass
    except Exception as e:
        st.error(f"Error al actualizar las estadísticas: {e}")

if webrtc_ctx.state.playing:
    plot_statistics()
