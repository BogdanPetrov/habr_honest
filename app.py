import html
from datetime import date, datetime, timedelta, timezone

import altair as alt
import pendulum
import polars as pl
import streamlit as st

ARTICLES_PATH = 'https://github.com/BogdanPetrov/habr_honest/raw/6166b168dcb4481e569745e1d3c2cbe7a96e4b76/habr_articles.txt.gz'
WORDS_PATH = 'https://github.com/BogdanPetrov/habr_honest/raw/6166b168dcb4481e569745e1d3c2cbe7a96e4b76/habr_word_data.txt.gz'
DATE_FROM = pendulum.parse('2020-09-07 00:00:00+00:00')
DATE_TO = pendulum.parse('2026-08-24 00:00:00+00:00')
DISPLAY_LIMIT = 2000
ALIGN_POS = 50
PUBLISHED_FMT = '%Y-%m-%dT%H:%M:%S%.fZ'
PERIODS = {
    '1 день': (1, None),
    '1 неделя': (7, '1w'),
    '2 недели': (14, '2w'),
    '4 недели': (28, '4w'),
}


@st.cache_data(show_spinner='Загрузка и предобработка данных...')
def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    df_articles_raw = pl.read_csv(ARTICLES_PATH, separator='\t')
    df_words_raw = pl.read_csv(WORDS_PATH, separator='\t')

    df_words = (
        df_words_raw
        .with_columns(
            exclude_word=(
                pl.col('word').str.contains('^не')
                | pl.col('word').str.contains('^бес')
                | (pl.len().over(pl.col('word')) <= 6)
            ),
            phrase_before=pl.concat_str(pl.col('word_before'), pl.lit(' '), pl.col('word')),
            phrase_after=pl.concat_str(pl.col('word'), pl.lit(' '), pl.col('word_after')),
        )
        .with_columns(
            exclude_phrase=(
                (pl.col('word') == 'по-честному')
                | (pl.coalesce(pl.col('phrase_after'), pl.lit('')) == 'честно говоря')
                | (pl.coalesce(pl.col('phrase_before'), pl.lit('')) == 'если честно')
                | (pl.coalesce(pl.col('phrase_before'), pl.lit('')) == 'будем честны')
            )
        )
    )

    author_first = (
        df_articles_raw
        .with_columns(pl.col('published').str.to_datetime(PUBLISHED_FMT, time_zone='UTC'))
        .group_by('author')
        .agg(pl.col('published').min().alias('author_first_published'))
    )

    author_article_counts = (
        df_articles_raw
        .with_columns(pl.col('published').str.to_datetime(PUBLISHED_FMT, time_zone='UTC'))
        .select('id', 'author', 'published')
        .with_columns(
            (
                pl.col('id').cum_count().over('author', order_by=['published', 'id']) - 1
            ).alias('author_article_count')
        )
        .select('id', 'author_article_count')
    )

    df_articles = (
        df_articles_raw
        .with_columns(pl.col('published').str.to_datetime(PUBLISHED_FMT, time_zone='UTC'))
        .with_columns(pl.col('published').dt.truncate('1w').alias('dt_week'))
        .with_columns(pl.col('published').dt.truncate('2w').alias('dt_week2'))
        .with_columns(pl.col('published').dt.truncate('4w').alias('dt_week4'))
        .with_columns(id_bucket=(pl.col('id') / 5000).round() * 5000)
        .with_columns(
            pl.coalesce(pl.col('location'), pl.lit('')).str.contains('/ru/companies/').alias('is_company')
        )
        .with_columns(pl.col('location').str.extract(r'companies/([^/]+)', 1).alias('company'))
        .join(author_first, on='author', how='left')
        .join(author_article_counts, on='id', how='left')
        .filter(pl.col('published') >= DATE_FROM)
        .filter(pl.col('published') < DATE_TO)
        .join(
            df_words.select('id').unique().with_columns(has_substr=pl.lit(True)),
            how='left',
            on='id',
        )
        .with_columns(pl.coalesce(pl.col('has_substr'), pl.lit(False)).alias('has_substr'))
    )

    return df_articles, df_words


def unique_csv_values(df: pl.DataFrame, col: str) -> list[str]:
    return (
        df
        .select(pl.col(col).fill_null('').str.split(',').alias('v'))
        .explode('v', empty_as_null=True)
        .with_columns(pl.col('v').str.strip_chars().str.strip_chars('"'))
        .filter(pl.col('v') != '')
        .unique()
        .sort('v')
        .get_column('v')
        .to_list()
    )


SCALE_LOW = -10
SCALE_HIGH = 50
ARTICLE_COUNT_HIGH = 10


def compressed_scale_options(low: int = SCALE_LOW, high: int = SCALE_HIGH, *, include_below: bool = True) -> list[str]:
    below = [f'< {low}'] if include_below else []
    return below + [str(n) for n in range(low, high + 1)] + [f'{high}+']


def scale_token_lo(token: str, low: int = SCALE_LOW, high: int = SCALE_HIGH) -> int | None:
    if token == f'< {low}':
        return None
    if token == f'{high}+':
        return high + 1
    return int(token)


def scale_token_hi(token: str, low: int = SCALE_LOW, high: int = SCALE_HIGH) -> int | None:
    if token == f'< {low}':
        return low - 1
    if token == f'{high}+':
        return None
    return int(token)


def compressed_range_slider(
    label: str,
    key: str,
    low: int = SCALE_LOW,
    high: int = SCALE_HIGH,
    *,
    include_below: bool = True,
    help_text: str | None = None,
) -> tuple[int | None, int | None]:
    options = compressed_scale_options(low, high, include_below=include_below)
    chosen = st.select_slider(
        label,
        options=options,
        value=(options[0], options[-1]),
        key=key,
        help=help_text or f'Ниже {low} и выше {high} — одним бакетом, внутри — по единице.',
    )
    token_from, token_to = chosen
    if options.index(token_from) > options.index(token_to):
        token_from, token_to = token_to, token_from
    return scale_token_lo(token_from, low, high), scale_token_hi(token_to, low, high)


def author_article_count_slider(label: str, key: str) -> tuple[int | None, int | None]:
    return compressed_range_slider(
        label,
        key,
        low=0,
        high=ARTICLE_COUNT_HIGH,
        include_below=False,
        help_text='Статей у автора до этой публикации (без учёта текущей). 10+ — десять и больше.',
    )


def unique_non_null(df: pl.DataFrame, col: str) -> list[str]:
    return (
        df
        .select(pl.col(col).cast(pl.Utf8))
        .drop_nulls()
        .filter(pl.col(col) != '')
        .unique()
        .sort(col)
        .get_column(col)
        .to_list()
    )


def to_utc_date(value) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).date()
    return value


def truncate_date(d: date, every: str) -> date:
    ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return to_utc_date(pl.Series([ts]).dt.truncate(every).item())


def window_range(view_date: date, period_label: str) -> tuple[date, date]:
    days, every = PERIODS[period_label]
    if every is None:
        return view_date, view_date
    start = truncate_date(view_date, every)
    return start, start + timedelta(days=days - 1)


def clamp_date(d: date, min_d: date, max_d: date) -> date:
    return max(min_d, min(max_d, d))


def shift_view_date(key: str, delta_days: int, min_d: date, max_d: date) -> None:
    st.session_state[key] = clamp_date(
        st.session_state[key] + timedelta(days=delta_days),
        min_d,
        max_d,
    )


def as_date(value) -> date | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return to_utc_date(value)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if value > 1e11:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()
        if value > 1e9:
            return datetime.fromtimestamp(value, tz=timezone.utc).date()
        return None
    if isinstance(value, list) and value:
        return as_date(value[0])
    date_method = getattr(value, 'date', None)
    if callable(date_method) and not isinstance(value, date):
        try:
            return as_date(date_method())
        except TypeError:
            pass
    return None


def week_from_chart_state(state) -> date | None:
    if state is None:
        return None
    selection = state.get('selection') if hasattr(state, 'get') else getattr(state, 'selection', None)
    if not selection:
        return None
    payload = selection.get('pick') if hasattr(selection, 'get') else getattr(selection, 'pick', None)
    return _week_from_payload(payload)


def date_range_from_brush(state) -> tuple[date, date] | None:
    if state is None:
        return None
    selection = state.get('selection') if hasattr(state, 'get') else getattr(state, 'selection', None)
    if not selection:
        return None
    payload = selection.get('brush') if hasattr(selection, 'get') else getattr(selection, 'brush', None)
    if not payload:
        return None
    raw = None
    if isinstance(payload, dict):
        raw = payload.get('week') or payload.get('неделя') or payload.get('x')
        if raw is None:
            for value in payload.values():
                if isinstance(value, list) and len(value) >= 2:
                    raw = value
                    break
    elif isinstance(payload, list) and len(payload) >= 2:
        raw = payload
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    start, end = as_date(raw[0]), as_date(raw[1])
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    return start, end


def _week_from_payload(payload) -> date | None:
    if payload is None or payload == {}:
        return None
    if isinstance(payload, dict):
        for key in ('week', 'неделя', 'x'):
            if key in payload:
                found = as_date(payload[key])
                if found is not None:
                    return found
        for value in payload.values():
            found = _week_from_payload(value)
            if found is not None:
                return found
        return None
    if isinstance(payload, list):
        for item in payload:
            found = _week_from_payload(item)
            if found is not None:
                return found
        return None
    return as_date(payload)


def apply_chart_week(event) -> None:
    week = week_from_chart_state(event)
    if week is None:
        return
    min_d = st.session_state.get('min_published')
    max_d = st.session_state.get('max_published')
    if min_d is None or max_d is None:
        return
    sig = week.isoformat()
    if st.session_state.get('chart_pick_sig') == sig:
        return
    st.session_state.chart_pick_sig = sig
    st.session_state.view_date_left = clamp_date(week, min_d, max_d)


def apply_chart_brush(event) -> None:
    rng = date_range_from_brush(event)
    if rng is None:
        return
    min_d = st.session_state.get('min_published')
    max_d = st.session_state.get('max_published')
    if min_d is None or max_d is None:
        return
    start, end = rng
    start = clamp_date(start, min_d, max_d)
    end = clamp_date(end, min_d, max_d)
    if start > end:
        start, end = end, start
    sig = f'{start.isoformat()}:{end.isoformat()}'
    if st.session_state.get('chart_brush_sig') == sig:
        return
    st.session_state.chart_brush_sig = sig
    st.session_state.chart_date_from = start
    st.session_state.chart_date_to = end


def share_chart(weekly: pl.DataFrame) -> alt.Chart:
    pdf = weekly.rename({'неделя': 'week', 'доля': 'share', 'всего': 'articles'}).to_pandas()
    pick = alt.selection_point(
        name='pick',
        fields=['week'],
        nearest=True,
        on='click',
        empty=False,
        toggle=False,
    )
    brush = alt.selection_interval(name='brush', encodings=['x'])
    x = alt.X('week:T', title='неделя')
    y_share = alt.Y(
        'share:Q',
        title='доля',
        axis=alt.Axis(titleColor='#6cb6ff', orient='left'),
    )
    y_share_ns = alt.Y('share:Q', axis=None)
    y_count = alt.Y(
        'articles:Q',
        title='статьи',
        axis=alt.Axis(titleColor='#999', orient='right'),
    )
    tooltips = [
        alt.Tooltip('week:T', title='неделя', format='%Y-%m-%d'),
        alt.Tooltip('share:Q', title='доля', format='.3f'),
        alt.Tooltip('articles:Q', title='статьи', format='d'),
    ]
    line = alt.Chart(pdf).mark_line(color='#6cb6ff').encode(x=x, y=y_share, tooltip=tooltips)
    points = alt.Chart(pdf).mark_circle(size=40, color='#6cb6ff').encode(x=x, y=y_share_ns, tooltip=tooltips)
    counts = alt.Chart(pdf).mark_area(opacity=0.18, color='#888').encode(x=x, y=y_count, tooltip=tooltips)
    hit = alt.Chart(pdf).mark_circle(size=400, opacity=0.01).encode(x=x, y=y_share_ns).add_params(pick)
    # Keep all points in the layer (opacity only) so independent y-scale keeps the full
    # share domain; transform_filter left a single point and floated it above the line.
    highlight = (
        alt.Chart(pdf)
        .mark_circle(size=90, filled=True, color='#ffe600')
        .encode(
            x=x,
            y=y_share_ns,
            opacity=alt.condition(pick, alt.value(1), alt.value(0)),
        )
    )
    return (
        alt.layer(line, points, counts, hit, highlight)
        .add_params(brush)
        .resolve_scale(y='independent')
        .properties(height=280)
    )


def filter_articles(
    df_articles: pl.DataFrame,
    date_from: date,
    date_to: date,
    company_mode: str,
    companies: list[str],
    author: str,
    hubs: list[str],
    labels: list[str],
    complexities: list[str],
    author_first_from: date | None = None,
    author_first_to: date | None = None,
    author_first_bounds: tuple[date, date] | None = None,
    author_article_count_from: int | None = None,
    author_article_count_to: int | None = None,
    rating_from: int | None = None,
    rating_to: int | None = None,
    karma_from: int | None = None,
    karma_to: int | None = None,
) -> pl.DataFrame:
    if date_from > date_to:
        date_from, date_to = date_to, date_from

    out = filter_by_published(df_articles, date_from, date_to)
    return apply_article_attr_filters(
        out,
        company_mode=company_mode,
        companies=companies,
        author=author,
        hubs=hubs,
        labels=labels,
        complexities=complexities,
        author_first_from=author_first_from,
        author_first_to=author_first_to,
        author_first_bounds=author_first_bounds,
        author_article_count_from=author_article_count_from,
        author_article_count_to=author_article_count_to,
        rating_from=rating_from,
        rating_to=rating_to,
        karma_from=karma_from,
        karma_to=karma_to,
    )


def filter_by_published(df_articles: pl.DataFrame, date_from: date, date_to: date) -> pl.DataFrame:
    if date_from > date_to:
        date_from, date_to = date_to, date_from
    start = pendulum.parse(f'{date_from.isoformat()} 00:00:00+00:00')
    end = pendulum.parse(f'{date_to.isoformat()} 00:00:00+00:00').add(days=1)
    return (
        df_articles
        .filter(pl.col('published') >= start)
        .filter(pl.col('published') < end)
    )


def filter_by_dt_week(df_articles: pl.DataFrame, date_from: date, date_to: date) -> pl.DataFrame:
    start = truncate_date(date_from, '1w')
    end = truncate_date(date_to, '1w')
    if start > end:
        start, end = end, start
    return (
        df_articles
        .filter(pl.col('dt_week').dt.date() >= start)
        .filter(pl.col('dt_week').dt.date() <= end)
    )


def apply_article_attr_filters(
    df_articles: pl.DataFrame,
    company_mode: str,
    companies: list[str],
    author: str,
    hubs: list[str],
    labels: list[str],
    complexities: list[str],
    author_first_from: date | None = None,
    author_first_to: date | None = None,
    author_first_bounds: tuple[date, date] | None = None,
    author_article_count_from: int | None = None,
    author_article_count_to: int | None = None,
    rating_from: int | None = None,
    rating_to: int | None = None,
    karma_from: int | None = None,
    karma_to: int | None = None,
) -> pl.DataFrame:
    out = df_articles

    bounds = author_first_bounds
    if (
        bounds is not None
        and author_first_from is not None
        and author_first_to is not None
        and (author_first_from != bounds[0] or author_first_to != bounds[1])
    ):
        first_from, first_to = author_first_from, author_first_to
        if first_from > first_to:
            first_from, first_to = first_to, first_from
        out = (
            out
            .filter(pl.col('author_first_published').dt.date() >= first_from)
            .filter(pl.col('author_first_published').dt.date() <= first_to)
        )

    if author_article_count_from is not None:
        out = out.filter(pl.col('author_article_count') >= author_article_count_from)
    if author_article_count_to is not None:
        out = out.filter(pl.col('author_article_count') <= author_article_count_to)

    if rating_from is not None:
        out = out.filter(pl.col('post_rating') >= rating_from)
    if rating_to is not None:
        out = out.filter(pl.col('post_rating') <= rating_to)

    if karma_from is not None:
        out = out.filter(pl.col('author_karma') >= karma_from)
    if karma_to is not None:
        out = out.filter(pl.col('author_karma') <= karma_to)

    if company_mode == 'company':
        out = out.filter(pl.col('is_company'))
    elif company_mode == 'personal':
        out = out.filter(~pl.col('is_company'))

    if companies:
        out = out.filter(pl.col('company').is_in(companies))

    author = author.strip()
    if author:
        needle = author.lower()
        out = out.filter(
            pl.col('author').fill_null('').str.to_lowercase().str.contains(needle, literal=True)
        )

    if hubs:
        out = out.filter(
            pl.col('hubs_str')
            .fill_null('')
            .str.split(',')
            .list.eval(pl.element().str.strip_chars())
            .list.set_intersection(pl.lit(hubs, dtype=pl.List(pl.Utf8)))
            .list.len()
            > 0
        )

    if labels:
        out = out.filter(
            pl.col('labels_str')
            .fill_null('')
            .str.split(',')
            .list.eval(pl.element().str.strip_chars().str.strip_chars('"'))
            .list.set_intersection(pl.lit(labels, dtype=pl.List(pl.Utf8)))
            .list.len()
            > 0
        )

    if complexities:
        out = out.filter(pl.col('complexity').cast(pl.Utf8).is_in(complexities))

    return out


def filter_words(
    df_words: pl.DataFrame,
    words: list[str],
    hide_exclude_word: bool,
    hide_exclude_phrase: bool,
) -> pl.DataFrame:
    out = df_words
    if words:
        out = out.filter(pl.col('word').is_in(words))
    if hide_exclude_word:
        out = out.filter(~pl.col('exclude_word'))
    if hide_exclude_phrase:
        out = out.filter(~pl.col('exclude_phrase'))
    return out


def matching_rows(df_articles: pl.DataFrame, df_words: pl.DataFrame) -> pl.DataFrame:
    pad = (pl.lit(ALIGN_POS) - pl.col('start_pos')).clip(lower_bound=0)
    return (
        df_articles
        .join(df_words, how='inner', on='id')
        .sort('id', 'idx')
        .with_columns(pl.col('text_around').str.replace_all('\xa0', ' '))
        .with_columns(
            pl.col('text_around')
            .str.pad_start(pl.col('text_around').str.len_chars() + pad)
            .alias('text_around')
        )
    )


def word_span(text: str, stem_at: int, word: str) -> tuple[int, int]:
    word_len = len(word)
    if not text or not word:
        return stem_at, stem_at
    text_lower = text.lower()
    word_lower = word.lower()
    for offset in range(word_len):
        start = stem_at - offset
        if start < 0:
            break
        if text_lower[start:start + word_len] == word_lower:
            return start, start + word_len
    return stem_at, min(stem_at + word_len, len(text))


def hits_html(df: pl.DataFrame) -> str:
    if df.height == 0:
        body = ''
    else:
        id_width = max(len(str(i)) for i in df.get_column('id').to_list())
        lines = []
        for row in df.iter_rows(named=True):
            text = row['text_around'] or ''
            word = row['word'] or ''
            start_pos = int(row['start_pos'])
            stem_at = start_pos + max(0, ALIGN_POS - start_pos)
            hit_from, hit_to = word_span(text, stem_at, word)
            before = html.escape(text[:hit_from])
            hit = html.escape(text[hit_from:hit_to])
            after = html.escape(text[hit_to:])
            id_raw = str(row['id'])
            pad = ' ' * (id_width - len(id_raw))
            id_link = (
                f'{pad}<a href="https://habr.com/ru/articles/{html.escape(id_raw)}"'
                ' target="_blank" rel="noopener noreferrer" style="color:#6cb6ff"'
                f'>{html.escape(id_raw)}</a>'
            )
            lines.append(
                f'{id_link} {before}<span style="background:#ffe600;color:#111">{hit}</span>{after}'
            )
        body = '\n'.join(lines)
    return (
        '<pre style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'
        '&quot;Liberation Mono&quot;,&quot;Courier New&quot;,monospace;font-size:13px;'
        'line-height:1.45;margin:0;white-space:pre;overflow-x:auto;">'
        f'{body}</pre>'
    )


def weekly_unique_articles(df: pl.DataFrame, name: str) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame({'неделя': [], name: []}, schema={'неделя': pl.Date, name: pl.UInt32})
    return (
        df
        .select('id', pl.col('dt_week').dt.date().alias('неделя'))
        .unique(subset='id')
        .group_by('неделя')
        .agg(pl.len().alias(name))
    )


def weekly_article_share(df_articles: pl.DataFrame, df_hits: pl.DataFrame) -> pl.DataFrame:
    if df_articles.height == 0:
        return pl.DataFrame(
            {'неделя': [], 'доля': [], 'всего': []},
            schema={'неделя': pl.Date, 'доля': pl.Float64, 'всего': pl.UInt32},
        )
    all_arts = weekly_unique_articles(df_articles, 'всего')
    hit_arts = weekly_unique_articles(df_hits, 'с_словом')
    start = all_arts.get_column('неделя').min()
    end = all_arts.get_column('неделя').max()
    weeks = pl.select(pl.date_range(start, end, interval='1w', eager=True).alias('неделя'))
    return (
        weeks
        .join(all_arts, on='неделя', how='left')
        .join(hit_arts, on='неделя', how='left')
        .with_columns(
            pl.col('всего').fill_null(0),
            pl.col('с_словом').fill_null(0),
        )
        .with_columns(
            pl.when(pl.col('всего') > 0)
            .then(pl.col('с_словом') / pl.col('всего'))
            .otherwise(0.0)
            .alias('доля')
        )
        .select('неделя', 'доля', 'всего')
        .sort('неделя')
    )


def render_date_panel(
    *,
    title: str,
    date_key: str,
    period_label: str,
    min_published: date,
    max_published: date,
    df_articles: pl.DataFrame,
    words_f: pl.DataFrame,
    company_mode: str,
    companies: list[str],
    author: str,
    hubs: list[str],
    labels: list[str],
    complexities: list[str],
    author_first_from: date,
    author_first_to: date,
    author_first_bounds: tuple[date, date],
    author_article_count_from: int | None,
    author_article_count_to: int | None,
    rating_from: int | None,
    rating_to: int | None,
    karma_from: int | None,
    karma_to: int | None,
) -> None:
    st.subheader(title)
    period_days, _every = PERIODS[period_label]
    view_date = st.session_state[date_key]
    date_from, date_to = window_range(view_date, period_label)
    prev_window = window_range(
        clamp_date(view_date - timedelta(days=period_days), min_published, max_published),
        period_label,
    )
    next_window = window_range(
        clamp_date(view_date + timedelta(days=period_days), min_published, max_published),
        period_label,
    )

    nav_prev, nav_date, nav_next = st.columns([1, 4, 1], vertical_alignment='bottom')
    with nav_prev:
        st.button(
            '←',
            key=f'{date_key}_prev',
            disabled=prev_window == (date_from, date_to),
            on_click=shift_view_date,
            args=(date_key, -period_days, min_published, max_published),
            width='stretch',
        )
    with nav_date:
        st.date_input(
            'Дата',
            min_value=min_published,
            max_value=max_published,
            key=date_key,
            label_visibility='collapsed',
        )
    with nav_next:
        st.button(
            '→',
            key=f'{date_key}_next',
            disabled=next_window == (date_from, date_to),
            on_click=shift_view_date,
            args=(date_key, period_days, min_published, max_published),
            width='stretch',
        )

    if date_from == date_to:
        st.caption(date_from.isoformat())
    else:
        st.caption(f'{date_from.isoformat()} — {date_to.isoformat()}')

    articles_f = filter_articles(
        df_articles,
        date_from=date_from,
        date_to=date_to,
        company_mode=company_mode,
        companies=companies,
        author=author,
        hubs=hubs,
        labels=labels,
        complexities=complexities,
        author_first_from=author_first_from,
        author_first_to=author_first_to,
        author_first_bounds=author_first_bounds,
        author_article_count_from=author_article_count_from,
        author_article_count_to=author_article_count_to,
        rating_from=rating_from,
        rating_to=rating_to,
        karma_from=karma_from,
        karma_to=karma_to,
    )
    result = matching_rows(articles_f, words_f)
    n_rows = result.height
    n_articles = result.select('id').n_unique() if n_rows else 0
    n_total = articles_f.select('id').n_unique() if articles_f.height else 0
    shown = result.head(DISPLAY_LIMIT)

    col_rows, col_arts, col_total = st.columns(3)
    col_rows.metric('Совпадений', n_rows)
    col_arts.metric('Статей', n_articles)
    col_total.metric('Всего статей', n_total)

    if n_rows > DISPLAY_LIMIT:
        st.warning(f'Показаны первые {DISPLAY_LIMIT} из {n_rows} строк')

    st.html(hits_html(shown))


def main() -> None:
    st.set_page_config(page_title='Word-O-Scope', layout='wide')
    st.markdown(
        '''
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                "Liberation Mono", "Courier New", monospace;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container,
        [data-testid="stMainBlockContainer"],
        section.stMain .block-container {
            padding-top: 1rem;
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )
    st.title('Word-O-Scope')
    st.caption('Вхождения из Habr с фильтрами по статьям и словам')

    df_articles, df_words = load_data()

    min_published = to_utc_date(df_articles.select(pl.col('published').min()).item())
    max_published = to_utc_date(df_articles.select(pl.col('published').max()).item())
    companies = unique_non_null(df_articles, 'company')
    hubs = unique_csv_values(df_articles, 'hubs_str')
    labels = unique_csv_values(df_articles, 'labels_str')
    complexities = unique_non_null(df_articles, 'complexity')
    words = unique_non_null(df_words, 'word')

    with st.sidebar:
        st.header('Статьи')
        company_mode_label = st.radio(
            'Тип блога',
            options=['Все', 'Только корпоративные', 'Только не корпоративные'],
            index=0,
        )
        company_mode = {
            'Все': 'all',
            'Только корпоративные': 'company',
            'Только не корпоративные': 'personal',
        }[company_mode_label]
        selected_companies = st.multiselect('Компания', options=companies)
        author = st.text_input(
            'Автор (логин, содержит)',
            help='Логин на Habr, не отображаемое имя. Регистр не важен.',
        )
        min_first = to_utc_date(df_articles.select(pl.col('author_first_published').min()).item())
        max_first = to_utc_date(df_articles.select(pl.col('author_first_published').max()).item())
        author_first_bounds = (min_first, max_first)
        author_first_from, author_first_to = st.slider(
            'Первая статья автора',
            min_value=min_first,
            max_value=max_first,
            value=(min_first, max_first),
            format='YYYY-MM-DD',
        )
        author_article_count_from, author_article_count_to = author_article_count_slider(
            'Статей у автора к моменту публикации',
            key='author_article_count_scale',
        )
        rating_from, rating_to = compressed_range_slider('Рейтинг статьи', key='rating_scale')
        karma_from, karma_to = compressed_range_slider('Карма автора', key='karma_scale')
        selected_hubs = st.multiselect('Хабы', options=hubs)
        selected_labels = st.multiselect('Лейблы', options=labels)
        selected_complexities = st.multiselect('Сложность', options=complexities)

        st.header('Слова')
        selected_words = st.multiselect('Слово', options=words)
        hide_exclude_word = st.checkbox('Скрыть exclude_word', value=False)
        hide_exclude_phrase = st.checkbox('Скрыть exclude_phrase', value=False)

    words_f = filter_words(
        df_words,
        words=selected_words,
        hide_exclude_word=hide_exclude_word,
        hide_exclude_phrase=hide_exclude_phrase,
    )

    if 'chart_date_from' not in st.session_state:
        st.session_state.chart_date_from = min_published
    if 'chart_date_to' not in st.session_state:
        st.session_state.chart_date_to = max_published
    if 'view_date_left' not in st.session_state:
        st.session_state.view_date_left = max_published
    if 'view_date_right' not in st.session_state:
        st.session_state.view_date_right = clamp_date(
            max_published - timedelta(days=7),
            min_published,
            max_published,
        )
    st.session_state.min_published = min_published
    st.session_state.max_published = max_published
    if 'share_chart' in st.session_state:
        apply_chart_brush(st.session_state.share_chart)

    chart_from_col, chart_to_col = st.columns(2)
    with chart_from_col:
        st.date_input(
            'График: дата с',
            min_value=min_published,
            max_value=max_published,
            key='chart_date_from',
        )
    with chart_to_col:
        st.date_input(
            'График: дата по',
            min_value=min_published,
            max_value=max_published,
            key='chart_date_to',
        )

    articles_chart = apply_article_attr_filters(
        filter_by_dt_week(
            df_articles,
            st.session_state.chart_date_from,
            st.session_state.chart_date_to,
        ),
        company_mode=company_mode,
        companies=selected_companies,
        author=author,
        hubs=selected_hubs,
        labels=selected_labels,
        complexities=selected_complexities,
        author_first_from=author_first_from,
        author_first_to=author_first_to,
        author_first_bounds=author_first_bounds,
        author_article_count_from=author_article_count_from,
        author_article_count_to=author_article_count_to,
        rating_from=rating_from,
        rating_to=rating_to,
        karma_from=karma_from,
        karma_to=karma_to,
    )
    hits_chart = matching_rows(articles_chart, words_f)
    weekly = weekly_article_share(articles_chart, hits_chart)
    if weekly.height:
        event = st.altair_chart(
            share_chart(weekly),
            key='share_chart',
            on_select='rerun',
            width='stretch',
        )
        apply_chart_week(event)
        st.caption(
            'Клик по точке выбирает неделю для левого списка. '
            'Выделение области мышкой задаёт диапазон дат графика.'
        )

    period_label = st.radio(
        'Интервал',
        options=list(PERIODS),
        index=0,
        horizontal=True,
        key='period_label',
    )

    panel_kwargs = dict(
        period_label=period_label,
        min_published=min_published,
        max_published=max_published,
        df_articles=df_articles,
        words_f=words_f,
        company_mode=company_mode,
        companies=selected_companies,
        author=author,
        hubs=selected_hubs,
        labels=selected_labels,
        complexities=selected_complexities,
        author_first_from=author_first_from,
        author_first_to=author_first_to,
        author_first_bounds=author_first_bounds,
        author_article_count_from=author_article_count_from,
        author_article_count_to=author_article_count_to,
        rating_from=rating_from,
        rating_to=rating_to,
        karma_from=karma_from,
        karma_to=karma_to,
    )

    left_col, right_col = st.columns(2, gap='large')
    with left_col:
        render_date_panel(
            title='Слева',
            date_key='view_date_left',
            **panel_kwargs,
        )
    with right_col:
        render_date_panel(
            title='Справа',
            date_key='view_date_right',
            **panel_kwargs,
        )


if __name__ == '__main__':
    main()
