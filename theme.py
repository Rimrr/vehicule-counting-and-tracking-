"""Thème moderne pour l'application Vehicle Counting (Streamlit).

Usage : en haut de app.py, après st.set_page_config(...), appeler inject_css().
"""
from html import escape

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{--ink:#0F1B2D;--paper:#F6F7F9;--surface:#FFFFFF;--line:#E2E6ED;--muted:#5B677A;
--road:#F0A500;--up:#0F8B8D;--down:#D9514E;--radius:14px}
.stApp{background:var(--paper);color:var(--ink);font-family:'Instrument Sans',system-ui,sans-serif}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
h1,h2,h3{font-family:'Bricolage Grotesque','Instrument Sans',sans-serif;letter-spacing:-0.02em;color:var(--ink)}
h2{font-size:1.5rem}
:focus-visible{outline:2px solid var(--road)!important;outline-offset:2px}

/* Barre latérale */
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:2px}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:9px 12px;border-radius:10px;width:100%;cursor:pointer;border-left:3px solid transparent}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:not(:has([data-testid="stMarkdownContainer"])){display:none}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:#F1F3F7}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:#EEF1F6;border-left-color:var(--road);font-weight:600}

/* Boutons, upload, onglets */
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font-weight:600;padding:.55rem 1.1rem;transition:border-color .15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--ink);color:var(--ink)}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--ink);border-color:var(--ink);color:#fff}
.stButton>button[kind="primary"]:hover{background:#1C2B45;color:#fff}
[data-testid="stFileUploaderDropzone"]{background:var(--surface);border:1.5px dashed #C3CAD6;border-radius:var(--radius)}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [aria-selected="true"]{color:var(--ink);font-weight:600}
.stTabs [data-baseweb="tab-highlight"]{background:var(--road)}
[data-testid="stImage"] img,video,[data-testid="stDataFrame"]{border-radius:12px}

/* Composants */
.vc-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:2.1rem;letter-spacing:-0.025em;line-height:1.15;margin:0}
.vc-sub{color:var(--muted);font-size:1.02rem;margin:.35rem 0 1.6rem;max-width:62ch}
.vc-hero{background:var(--ink);color:#fff;border-radius:20px;padding:30px 34px;margin-bottom:18px}
.vc-hero-row{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.vc-hero-num{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:4.6rem;line-height:1;font-variant-numeric:tabular-nums}
.vc-hero-txt{color:#B9C2D1;font-size:1.05rem}
.vc-bar{display:flex;height:10px;border-radius:99px;overflow:hidden;background:#26354D;margin:22px 0 12px}
.vc-bar span{display:block;height:100%}
.vc-legend{display:flex;gap:26px;flex-wrap:wrap;color:#D5DBE6;font-size:.95rem}
.vc-legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
.vc-legend b{color:#fff;font-variant-numeric:tabular-nums}
.vc-strip{display:flex;flex-wrap:wrap;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);margin-bottom:22px}
.vc-stat{flex:1 1 140px;padding:16px 22px;border-right:1px solid var(--line)}
.vc-stat:last-child{border-right:none}
.vc-stat small{display:block;color:var(--muted);font-size:.85rem;margin-bottom:4px}
.vc-stat strong{font-family:'Bricolage Grotesque',sans-serif;font-size:1.7rem;font-variant-numeric:tabular-nums}
.vc-callout{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--road);border-radius:10px;padding:14px 18px;margin:12px 0;font-size:.97rem}
.vc-callout.info{border-left-color:var(--up)}
@media (max-width:640px){.vc-hero{padding:22px}.vc-hero-num{font-size:3.4rem}.vc-stat{border-right:none;border-bottom:1px solid var(--line)}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<p class="vc-sub">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<h1 class="vc-title">{escape(title)}</h1>{sub}', unsafe_allow_html=True)


def hero_total(total: int, up: int, down: int) -> None:
    """Bloc principal : total de véhicules et répartition Up / Down."""
    base = max(up + down, 1)
    bar = (f'<span style="width:{up / base * 100:.1f}%;background:var(--up)"></span>'
           f'<span style="width:{down / base * 100:.1f}%;background:var(--down)"></span>') if up + down else ""
    st.markdown(
        '<section class="vc-hero"><div class="vc-hero-row">'
        f'<div class="vc-hero-num">{int(total)}</div>'
        '<div class="vc-hero-txt">véhicules ont franchi la ligne</div></div>'
        f'<div class="vc-bar">{bar}</div>'
        '<div class="vc-legend">'
        f'<span><i style="background:var(--up)"></i><b>{int(up)}</b> vers le haut</span>'
        f'<span><i style="background:var(--down)"></i><b>{int(down)}</b> vers le bas</span>'
        '</div></section>',
        unsafe_allow_html=True,
    )


def stat_strip(items: list[tuple[str, object]]) -> None:
    """Bandeau de statistiques : [("Voitures", 6), ("Motos", 2), ...]."""
    cells = "".join(f'<div class="vc-stat"><small>{escape(str(k))}</small><strong>{escape(str(v))}</strong></div>'
                    for k, v in items)
    st.markdown(f'<div class="vc-strip">{cells}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "warn") -> None:
    """Message d'information (kind='info') ou d'attention (kind='warn')."""
    st.markdown(f'<div class="vc-callout {escape(kind)}">{escape(text)}</div>', unsafe_allow_html=True)


def style_fig(fig):
    """Applique le style du thème à un graphique Plotly."""
    fig.update_layout(
        font=dict(family="Instrument Sans, system-ui, sans-serif", color="#0F1B2D"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=16), x=0),
        margin=dict(t=56, b=16, l=8, r=8),
    )
    fig.update_xaxes(gridcolor="#E2E6ED", zeroline=False)
    fig.update_yaxes(gridcolor="#E2E6ED", zeroline=False)
    return fig