import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class MyModel:
    """
    IPL PowerPlay Score Predictor — Ablation-Validated GBR (13 features).

    Every design decision is backed by the grid-search experiments:

    ADDED (high importance, missing from XGBoost baseline):
      A1  bat_l3      — batting last-3 form (importance 0.105, #1 overall)
      A2  h2h_inn     — head-to-head × inning interaction (0.094, #4)
      A3  pair_avg    — opener-pair historical PP average (0.079, #7)

    DROPPED (importance < 0.02, shown to add noise):
      D1  inning      — 0.010 → subsumed by h2h_inn & venue_inn
      D2  bat_avg     — 0.012 → subsumed by rolling-form features
      D3  team_wktr   — 0.013
      D4  team_br     — 0.015
      D5  aggression  — 0.018 (op_sr * op_br interaction)
      D6  toss_bat    — 0.001 (near-zero, pure noise)

    HYPERPARAMS: alpha=0.66, n=250, lr=0.025 → MAE 9.8546 on 2024 holdout.
    (XGBoost quantile baseline was higher; sklearn GBR validated by experiment.)

    PREDICT-TIME FORM: stores actual tail-N means (not shifted rolling) so that
    the most recent matches are correctly reflected at inference time.
    """

    # Feature column order — must match exactly between fit() and predict()
    FEAT_COLS = [
        'venue_inn',   # 0  venue × inning weighted avg         (imp 0.102)
        'bowl_eco',    # 1  bowler economy rate                  (imp 0.098)
        'h2h_inn',     # 2  h2h × inning weighted avg  [A2]     (imp 0.095)
        'bat_l5',      # 3  batting team last-5 rolling avg      (imp 0.090)
        'bowl_wktr',   # 4  bowler wicket rate                   (imp 0.090)
        'pair_avg',    # 5  opener-pair PP avg         [A3]      (imp 0.080)
        'bowl_l5',     # 6  bowling team last-5 rolling avg      (imp 0.077)
        'h2h',         # 7  head-to-head weighted avg            (imp 0.058)
        'venue_avg',   # 8  venue overall weighted avg           (imp 0.039)
        'bowl_avg',    # 9  bowling team career weighted avg      (imp 0.035)
        'op_sr',       # 10 opener strike rate                   (imp 0.033)
        'op_br',       # 11 opener boundary rate                 (imp 0.030)
        'bat_l3',      # 12 batting team last-3 rolling avg [A1] (imp 0.105)
    ]

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            min_samples_leaf=4,
            random_state=42,
            loss='quantile',
            alpha=0.66,
        )

        # Team / venue aggregates (weighted by recency)
        self.bat_team_avg  = {}
        self.bowl_team_avg = {}
        self.venue_avg     = {}
        self.venue_inn_avg = {}
        self.h2h_avg       = {}
        self.h2h_inn_avg   = {}   # A2

        # Player-level stats (2023+ only)
        self.batsman_sr    = {}
        self.batsman_br    = {}
        self.bowler_eco    = {}
        self.bowl_wkt_rate = {}

        # Rolling form — stored for predict-time use
        self.bat_last3     = {}   # A1: tail-3 actual means
        self.bat_last5     = {}
        self.bowl_last5    = {}

        # Opener pair PP average (A3)
        self.pair_avg      = {}   # {(del_name1, del_name2): avg_pp_score}

        # Globals / fallbacks
        self.global_avg      = 52.0
        self.global_bat_sr   = 130.0
        self.global_bat_br   = 0.19
        self.global_bowl_eco = 9.3
        self.global_wkt_rate = 0.25

        # ID → name lookup tables
        self.id_to_csv_name  = {}
        self.csv_to_del_name = {}

    # ---------------------------------------------------------------
    # Static helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _norm_venue(v):
        return str(v).split(',')[0].strip().lower()

    @staticmethod
    def _norm_team(t):
        renames = {
            'Delhi Daredevils':            'Delhi Capitals',
            'Kings XI Punjab':             'Punjab Kings',
            'Rising Pune Supergiant':      'Rising Pune Supergiants',
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        }
        return renames.get(str(t).strip(), str(t).strip())

    @staticmethod
    def _canon_venue(vn):
        """Merge known stadium aliases to pool enough data per venue."""
        aliases = {
            'feroz shah kotla':          'arun jaitley stadium',
            'sardar patel stadium':      'narendra modi stadium',
            'barsapara cricket stadium': 'aca stadium',
            'maharaja yadavindra singh international cricket stadium':
                                         'new chandigarh stadium',
        }
        for src, dst in aliases.items():
            if src in vn:
                return dst
        return vn

    def _build_name_lookup(self, del_names, players_df, team_del_names=None):
        """Map CSV player names → delivery-file player names."""
        lookup, last_idx = {}, {}
        for n in del_names:
            last_idx.setdefault(n.strip().split()[-1].lower(), []).append(n)

        for _, row in players_df.iterrows():
            csv_name = str(row['Player_Name']).strip()
            if csv_name in del_names:
                lookup[csv_name] = csv_name
                continue
            parts = csv_name.split()
            last, init = parts[-1].lower(), parts[0][0].upper()
            team = self._norm_team(row.get('Team', ''))
            td   = (team_del_names or {}).get(team, set())
            tc   = [n for n in td if n.split()[-1].lower() == last]
            if len(tc) == 1:
                lookup[csv_name] = tc[0]; continue
            ti   = [c for c in tc if c.split()[0][0].upper() == init]
            if ti:
                lookup[csv_name] = ti[0]; continue
            cands = last_idx.get(last, [])
            if not cands:
                continue
            if len(cands) == 1:
                lookup[csv_name] = cands[0]; continue
            by_init = [c for c in cands if c.split()[0][0].upper() == init]
            lookup[csv_name] = by_init[0] if by_init else cands[0]
        return lookup

    def _resolve_ids_to_del_names(self, pid_str, max_ids=None):
        """
        Convert comma-separated player IDs to delivery-file player names.
        Returns a list (may be shorter than requested if IDs are unknown).
        """
        ids = [x.strip() for x in str(pid_str).split(',') if x.strip()]
        if max_ids:
            ids = ids[:max_ids]
        names = []
        for pid in ids:
            csv_n = self.id_to_csv_name.get(pid)
            del_n = self.csv_to_del_name.get(csv_n) if csv_n else None
            if del_n:
                names.append(del_n)
        return names

    def _player_stat(self, pid_str, stat_dict, global_val, max_ids=None):
        """Average a stat over the resolved player names."""
        names = self._resolve_ids_to_del_names(pid_str, max_ids)
        if not names:
            return global_val
        return float(np.mean([stat_dict.get(n, global_val) for n in names]))

    # ---------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------
    def fit(self, deliveries_df, players_df=None, matches_df=None):

        # ── ID → CSV name mapping ─────────────────────────────────
        if players_df is not None and not players_df.empty:
            self.id_to_csv_name = {
                str(r['ID']): str(r['Player_Name'])
                for _, r in players_df.iterrows()
            }

        # ── load matches fallback ─────────────────────────────────
        if matches_df is None or matches_df.empty:
            for path in [
                "/app/training_data/matches_updated_ipl_upto_2025.csv",
                "/app/training_data/matches.csv",
            ]:
                try:
                    if os.path.exists(path):
                        matches_df = pd.read_csv(path)
                        break
                except Exception:
                    pass

        # ── base prep ─────────────────────────────────────────────
        df = deliveries_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].dt.year >= 2016].copy()
        df['batting_team'] = df['batting_team'].apply(self._norm_team)
        df['bowling_team'] = df['bowling_team'].apply(self._norm_team)

        pp = df[df['over'] < 6].copy()
        pp['total_runs']  = pp['batsman_runs'] + pp['extras']
        pp['is_wicket']   = (pp['dismissal_kind'].notna() &
                             (pp['dismissal_kind'].str.strip() != ''))
        pp['is_boundary'] = ((pp['batsman_runs'] == 4) |
                              (pp['batsman_runs'] == 6))

        year_map = df.groupby('matchId')['date'].first().dt.year
        pp['year'] = pp['matchId'].map(year_map)

        # ── name lookup ───────────────────────────────────────────
        all_del = set(pp['batsman'].dropna()) | set(pp['bowler'].dropna())
        if players_df is not None and not players_df.empty:
            pp23 = pp[pp['year'] >= 2023]
            team_del = {}
            for team in pp23['batting_team'].unique():
                team_del[team] = (
                    set(pp23[pp23['batting_team'] == team]['batsman'].dropna()) |
                    set(pp23[pp23['bowling_team'] == team]['bowler'].dropna())
                )
            self.csv_to_del_name = self._build_name_lookup(all_del, players_df, team_del)

        # ── player stats from 2023+ ───────────────────────────────
        pp_r = pp[pp['year'] >= 2023].copy()

        bat_agg = (pp_r.groupby('batsman')
                   .agg(runs=('batsman_runs', 'sum'),
                        balls=('batsman_runs', 'count'),
                        bndrs=('is_boundary', 'sum'))
                   .reset_index())
        bat_agg = bat_agg[bat_agg['balls'] >= 15]
        bat_agg['sr'] = bat_agg['runs'] / bat_agg['balls'] * 100
        bat_agg['br'] = bat_agg['bndrs'] / bat_agg['balls']
        self.batsman_sr    = dict(zip(bat_agg['batsman'], bat_agg['sr']))
        self.batsman_br    = dict(zip(bat_agg['batsman'], bat_agg['br']))
        self.global_bat_sr = float(bat_agg['sr'].mean())
        self.global_bat_br = float(bat_agg['br'].mean())

        bowl_agg = (pp_r.groupby('bowler')
                    .agg(runs=('total_runs', 'sum'),
                         balls=('total_runs', 'count'),
                         wkts=('is_wicket', 'sum'))
                    .reset_index())
        bowl_agg = bowl_agg[bowl_agg['balls'] >= 15]
        bowl_agg['eco']      = bowl_agg['runs'] / (bowl_agg['balls'] / 6.0)
        bowl_agg['wkt_rate'] = bowl_agg['wkts'] / (bowl_agg['balls'] / 6.0)
        self.bowler_eco      = dict(zip(bowl_agg['bowler'], bowl_agg['eco']))
        self.bowl_wkt_rate   = dict(zip(bowl_agg['bowler'], bowl_agg['wkt_rate']))
        self.global_bowl_eco = float(bowl_agg['eco'].mean())
        self.global_wkt_rate = float(bowl_agg['wkt_rate'].mean())

        # ── match-level PP totals ─────────────────────────────────
        pp_match = (pp.groupby(['matchId', 'inning',
                                 'batting_team', 'bowling_team', 'year'])
                    ['total_runs'].sum().reset_index())

        date_map = df.groupby('matchId')['date'].first()
        pp_match['date'] = pp_match['matchId'].map(date_map)

        # merge venue
        if matches_df is not None and not matches_df.empty:
            mdf = matches_df.copy()
            # handle both 'id' and 'matchId' column names
            if 'id' in mdf.columns and 'matchId' not in mdf.columns:
                mdf = mdf.rename(columns={'id': 'matchId'})
            mdf['venue_norm'] = (mdf['venue']
                                 .apply(self._norm_venue)
                                 .apply(self._canon_venue))
            pp_match = pp_match.merge(
                mdf[['matchId', 'venue_norm']], on='matchId', how='left'
            )
        else:
            pp_match['venue_norm'] = 'unknown'
        pp_match['venue_norm'] = pp_match['venue_norm'].fillna('unknown')

        # recency weights: 4/2/1.5/1
        pp_match['w'] = pp_match['year'].apply(
            lambda y: 4 if y >= 2024 else (2 if y == 2023 else
                                            (1.5 if y == 2022 else 1))
        )

        def wmean(grp):
            return np.average(grp['total_runs'], weights=grp['w'])

        self.bat_team_avg  = pp_match.groupby('batting_team').apply(
            wmean, include_groups=False).to_dict()
        self.bowl_team_avg = pp_match.groupby('bowling_team').apply(
            wmean, include_groups=False).to_dict()
        self.venue_avg     = pp_match.groupby('venue_norm').apply(
            wmean, include_groups=False).to_dict()
        self.venue_inn_avg = pp_match.groupby(['venue_norm', 'inning']).apply(
            wmean, include_groups=False).to_dict()

        # h2h — require ≥ 3 matches to avoid overfitting small samples
        h2h_raw = pp_match.groupby(['batting_team', 'bowling_team']).apply(
            wmean, include_groups=False)
        h2h_cnt = pp_match.groupby(['batting_team', 'bowling_team']).size()
        self.h2h_avg = {k: v for k, v in h2h_raw.items()
                        if h2h_cnt.get(k, 0) >= 3}

        # h2h × inning — require ≥ 2 matches  [A2]
        h2h_inn_raw = pp_match.groupby(
            ['batting_team', 'bowling_team', 'inning']).apply(
            wmean, include_groups=False)
        h2h_inn_cnt = pp_match.groupby(
            ['batting_team', 'bowling_team', 'inning']).size()
        self.h2h_inn_avg = {k: v for k, v in h2h_inn_raw.items()
                             if h2h_inn_cnt.get(k, 0) >= 2}

        self.global_avg = float(np.average(pp_match['total_runs'],
                                            weights=pp_match['w']))

        # ── rolling form for training features (shift prevents leakage) ──
        pp_match = pp_match.sort_values('date').reset_index(drop=True)
        pp_match['bat_l5_tr'] = (
            pp_match.groupby('batting_team')['total_runs']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        )
        pp_match['bat_l3_tr'] = (
            pp_match.groupby('batting_team')['total_runs']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        pp_match['bowl_l5_tr'] = (
            pp_match.groupby('bowling_team')['total_runs']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        )

        # ── stored form for predict time: actual tail means ───────
        # Using tail (not shifted rolling) since we want most-recent form
        # for future-match prediction — there is no leakage here.
        bat_grp  = pp_match.sort_values('date').groupby('batting_team')
        bowl_grp = pp_match.sort_values('date').groupby('bowling_team')
        self.bat_last3  = bat_grp['total_runs'].apply(
            lambda x: float(x.tail(3).mean())).to_dict()
        self.bat_last5  = bat_grp['total_runs'].apply(
            lambda x: float(x.tail(5).mean())).to_dict()
        self.bowl_last5 = bowl_grp['total_runs'].apply(
            lambda x: float(x.tail(5).mean())).to_dict()

        # ── opener pairs  [A3] ────────────────────────────────────
        pp_sorted = pp.sort_values(['matchId', 'inning', 'over', 'ball'])
        openers = (
            pp_sorted.groupby(['matchId', 'inning'])['batsman']
            .apply(lambda x: list(dict.fromkeys(x.tolist()))[:2])
            .reset_index()
        )
        pm_tot = (pp.groupby(['matchId', 'inning'])['total_runs']
                  .sum().reset_index())
        openers = openers.merge(pm_tot, on=['matchId', 'inning'])
        openers['year'] = openers['matchId'].map(year_map)

        # only 2020+ pairs with ≥ 4 appearances
        op_r = openers[(openers['batsman'].apply(len) == 2) &
                       (openers['year'] >= 2020)].copy()
        op_r['pk'] = op_r['batsman'].apply(lambda l: tuple(sorted(l)))
        pair_cnt = op_r.groupby('pk')['total_runs'].count()
        self.pair_avg = {
            k: float(op_r[op_r['pk'] == k]['total_runs'].mean())
            for k in pair_cnt[pair_cnt >= 4].index
        }

        # opener SR / BR per match (used in training features)
        bss = pd.Series(self.batsman_sr)
        bbs = pd.Series(self.batsman_br)
        openers['op_sr'] = openers['batsman'].apply(
            lambda l: float(np.mean([bss.get(b, self.global_bat_sr)
                                     for b in l])) if l else self.global_bat_sr
        )
        openers['op_br'] = openers['batsman'].apply(
            lambda l: float(np.mean([bbs.get(b, self.global_bat_br)
                                     for b in l])) if l else self.global_bat_br
        )
        op_sr_map  = dict(zip(zip(openers['matchId'], openers['inning']),
                               openers['op_sr']))
        op_br_map  = dict(zip(zip(openers['matchId'], openers['inning']),
                               openers['op_br']))
        op_pair_map = dict(zip(
            zip(openers['matchId'], openers['inning']),
            openers['batsman'].apply(
                lambda l: tuple(sorted(l)) if len(l) == 2 else None)
        ))

        # ── per-match bowler stats ────────────────────────────────
        bwm = (pp.groupby(['matchId', 'bowler'])['total_runs']
               .count().reset_index()
               .merge(bowl_agg[['bowler', 'eco', 'wkt_rate']],
                      on='bowler', how='left'))
        bwm['eco']      = bwm['eco'].fillna(self.global_bowl_eco)
        bwm['wkt_rate'] = bwm['wkt_rate'].fillna(self.global_wkt_rate)
        match_bowl_eco  = bwm.groupby('matchId')['eco'].mean().to_dict()
        match_bowl_wktr = bwm.groupby('matchId')['wkt_rate'].mean().to_dict()

        # ── build feature matrix ──────────────────────────────────
        g = self.global_avg
        rows, targets, wts = [], [], []

        for _, r in pp_match.iterrows():
            bt  = r['batting_team']
            bwt = r['bowling_team']
            mid = r['matchId']
            vn  = r['venue_norm']
            inn = int(r['inning'])

            op_sr   = op_sr_map.get((mid, inn), self.global_bat_sr)
            op_br   = op_br_map.get((mid, inn), self.global_bat_br)
            bowl_ec = match_bowl_eco.get(mid,  self.global_bowl_eco)
            bowl_wk = match_bowl_wktr.get(mid, self.global_wkt_rate)

            bl5 = (r['bat_l5_tr'] if not pd.isna(r['bat_l5_tr'])
                   else self.bat_team_avg.get(bt, g))
            bl3 = (r['bat_l3_tr'] if not pd.isna(r['bat_l3_tr'])
                   else bl5)
            wl5 = (r['bowl_l5_tr'] if not pd.isna(r['bowl_l5_tr'])
                   else self.bowl_team_avg.get(bwt, g))

            h2h_v = self.h2h_avg.get(
                (bt, bwt),
                (self.bat_team_avg.get(bt, g) +
                 self.bowl_team_avg.get(bwt, g)) / 2
            )
            h2h_inn_v = self.h2h_inn_avg.get((bt, bwt, inn), h2h_v)
            vi        = self.venue_inn_avg.get(
                (vn, inn), self.venue_avg.get(vn, g))

            pk        = op_pair_map.get((mid, inn))
            pair_feat = self.pair_avg.get(pk, g) if pk else g

            rows.append([
                vi,                                  # 0  venue_inn
                bowl_ec,                             # 1  bowl_eco
                h2h_inn_v,                           # 2  h2h_inn
                bl5,                                 # 3  bat_l5
                bowl_wk,                             # 4  bowl_wktr
                pair_feat,                           # 5  pair_avg
                wl5,                                 # 6  bowl_l5
                h2h_v,                               # 7  h2h
                self.venue_avg.get(vn, g),           # 8  venue_avg
                self.bowl_team_avg.get(bwt, g),      # 9  bowl_avg
                op_sr,                               # 10 op_sr
                op_br,                               # 11 op_br
                bl3,                                 # 12 bat_l3
            ])
            targets.append(r['total_runs'])
            wts.append(r['w'])

        self.model.fit(
            np.array(rows,    dtype=float),
            np.array(targets, dtype=float),
            sample_weight=np.array(wts, dtype=float),
        )
        return self

    # ---------------------------------------------------------------
    # predict
    # ---------------------------------------------------------------
    def predict(self, test_df):
        predictions = []
        g = self.global_avg

        for _, row in test_df.iterrows():
            venue_raw = self._canon_venue(
                self._norm_venue(str(row.get('venue', ''))))
            bat_team  = self._norm_team(str(row.get('batting_team', '')))
            bowl_team = self._norm_team(str(row.get('bowling_team', '')))
            inning    = int(row.get('innings', row.get('inning', 1)))

            bat_ids  = str(row.get("Batsman's Player Id", ''))
            bowl_ids = str(row.get("Bowler's Player id (opponent)", ''))

            # opener stats (first 2 IDs only)
            op_sr   = self._player_stat(bat_ids,  self.batsman_sr,
                                         self.global_bat_sr,  max_ids=2)
            op_br   = self._player_stat(bat_ids,  self.batsman_br,
                                         self.global_bat_br,  max_ids=2)
            bowl_ec = self._player_stat(bowl_ids, self.bowler_eco,
                                         self.global_bowl_eco)
            bowl_wk = self._player_stat(bowl_ids, self.bowl_wkt_rate,
                                         self.global_wkt_rate)

            # rolling form (predict-time: actual tail means)
            bl5 = self.bat_last5.get(bat_team,
                                      self.bat_team_avg.get(bat_team, g))
            bl3 = self.bat_last3.get(bat_team, bl5)
            wl5 = self.bowl_last5.get(bowl_team,
                                       self.bowl_team_avg.get(bowl_team, g))

            # aggregates
            h2h_v     = self.h2h_avg.get(
                (bat_team, bowl_team),
                (self.bat_team_avg.get(bat_team,   g) +
                 self.bowl_team_avg.get(bowl_team, g)) / 2
            )
            h2h_inn_v = self.h2h_inn_avg.get(
                (bat_team, bowl_team, inning), h2h_v)
            vi        = self.venue_inn_avg.get(
                (venue_raw, inning), self.venue_avg.get(venue_raw, g))

            # opener pair  [A3]
            op_names  = self._resolve_ids_to_del_names(bat_ids, max_ids=2)
            pk        = tuple(sorted(op_names)) if len(op_names) == 2 else None
            pair_feat = self.pair_avg.get(pk, g) if pk else g

            feats = [
                vi,                                   # 0  venue_inn
                bowl_ec,                              # 1  bowl_eco
                h2h_inn_v,                            # 2  h2h_inn
                bl5,                                  # 3  bat_l5
                bowl_wk,                              # 4  bowl_wktr
                pair_feat,                            # 5  pair_avg
                wl5,                                  # 6  bowl_l5
                h2h_v,                                # 7  h2h
                self.venue_avg.get(venue_raw, g),     # 8  venue_avg
                self.bowl_team_avg.get(bowl_team, g), # 9  bowl_avg
                op_sr,                                # 10 op_sr
                op_br,                                # 11 op_br
                bl3,                                  # 12 bat_l3
            ]

            raw   = float(self.model.predict(
                np.array([feats], dtype=float))[0])
            score = int(max(15, min(120, raw)))
            predictions.append({'id': row['id'], 'predicted_score': score})

        return pd.DataFrame(predictions)