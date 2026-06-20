import React, { useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, ScrollView, ActivityIndicator, StatusBar,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, IRI_COLORS, IRI_LABELS } from '../config/env';
import { observationService } from '../services/observationService';
import { authService } from '../services/authService';

interface Stats {
  total:    number;
  good:     number;
  moderate: number;
  bad:      number;
  potholes: number;
  avgIri:   number;
}

function compute(data: any[]): Stats {
  const stats: Stats = { total: data.length, good: 0, moderate: 0, bad: 0, potholes: 0, avgIri: 0 };
  let sum = 0;
  data.forEach(d => {
    const iri: number = d.iriScore ?? 0;
    sum += iri;
    if (iri < 1.5)      stats.good++;
    else if (iri < 2.5) stats.moderate++;
    else                stats.bad++;
    if (d.hasPothole)   stats.potholes++;
  });
  stats.avgIri = data.length ? sum / data.length : 0;
  return stats;
}

const StatCard = ({ label, value, color, sub }: { label: string; value: string | number; color: string; sub?: string }) => (
  <View style={[styles.card, { borderTopColor: color }]}>
    <Text style={[styles.cardValue, { color }]}>{value}</Text>
    <Text style={styles.cardLabel}>{label}</Text>
    {sub ? <Text style={styles.cardSub}>{sub}</Text> : null}
  </View>
);

const Bar = ({ label, count, total, color }: { label: string; count: number; total: number; color: string }) => {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <View style={styles.barRow}>
      <View style={styles.barLabelWrap}>
        <View style={[styles.barDot, { backgroundColor: color }]} />
        <Text style={styles.barLabel}>{label}</Text>
      </View>
      <View style={styles.barTrack}>
        <View style={[styles.barFill, { width: `${pct}%` as any, backgroundColor: color }]} />
      </View>
      <Text style={styles.barCount}>{count}</Text>
    </View>
  );
};

export const StatsScreen = () => {
  const [stats,   setStats]   = useState<Stats | null>(null);
  const [user,    setUser]    = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const [history, userData] = await Promise.all([
        observationService.getHistory(),
        authService.getUser(),
      ]);
      setStats(compute(history));
      setUser(userData);
    } catch (e) {
      console.warn(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  const s = stats!;

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.scroll}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Stats</Text>
        <Text style={styles.headerSub}>
          {user?.isAnonymous ? 'Guest user' : (user?.name ?? 'Contributor')}
        </Text>
      </View>

      {/* Summary cards */}
      <View style={styles.grid}>
        <StatCard label="Total Reports"    value={s.total}             color={COLORS.primary}  />
        <StatCard label="Potholes Found"   value={s.potholes}          color='#F44336' />
        <StatCard label="Avg IRI"          value={s.avgIri.toFixed(2)} color='#FFC107' sub="m/km" />
        <StatCard label="Good Roads"       value={s.good}              color={IRI_COLORS.green} />
      </View>

      {/* Distribution */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Road Quality Distribution</Text>
        <View style={styles.sectionCard}>
          <Bar label={IRI_LABELS.green}  count={s.good}     total={s.total} color={IRI_COLORS.green} />
          <Bar label={IRI_LABELS.yellow} count={s.moderate} total={s.total} color={IRI_COLORS.yellow} />
          <Bar label={IRI_LABELS.orange} count={s.bad}      total={s.total} color={IRI_COLORS.orange} />
        </View>
      </View>

      {/* Contribution info */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About Your Contribution</Text>
        <View style={styles.sectionCard}>
          <Text style={styles.infoText}>
            Every observation you submit helps build a real-time road quality map for Delhi NCR.
            Your data is anonymised and aggregated with other riders to produce accurate, crowd-sourced road quality scores.
          </Text>
        </View>
      </View>

    </ScrollView>
  );
};

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: COLORS.bg },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: COLORS.bg },
  scroll: { paddingBottom: 40 },

  header: {
    paddingHorizontal: 20,
    paddingTop:        60,
    paddingBottom:     20,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  headerTitle: { color: COLORS.text,         fontSize: 22, fontWeight: '700' },
  headerSub:   { color: COLORS.textSecondary, fontSize: 13, marginTop: 2 },

  grid: {
    flexDirection:   'row',
    flexWrap:        'wrap',
    padding:         12,
    gap:             10,
  },
  card: {
    flex:            1,
    minWidth:        '45%',
    backgroundColor: COLORS.surface,
    borderRadius:    14,
    padding:         16,
    borderTopWidth:  3,
    borderWidth:     1,
    borderColor:     COLORS.border,
  },
  cardValue: { fontSize: 28, fontWeight: '700' },
  cardLabel: { color: COLORS.textSecondary, fontSize: 12, marginTop: 4 },
  cardSub:   { color: COLORS.textMuted,     fontSize: 10 },

  section:     { paddingHorizontal: 16, marginTop: 8 },
  sectionTitle:{ color: COLORS.textSecondary, fontSize: 12, fontWeight: '700', letterSpacing: 1, marginBottom: 10, paddingHorizontal: 4 },
  sectionCard: {
    backgroundColor: COLORS.surface,
    borderRadius:    14,
    padding:         16,
    borderWidth:     1,
    borderColor:     COLORS.border,
    gap:             14,
  },

  barRow:      { flexDirection: 'row', alignItems: 'center', gap: 10 },
  barLabelWrap:{ flexDirection: 'row', alignItems: 'center', width: 80, gap: 6 },
  barDot:      { width: 8, height: 8, borderRadius: 4 },
  barLabel:    { color: COLORS.textSecondary, fontSize: 12 },
  barTrack:    { flex: 1, height: 6, backgroundColor: COLORS.surfaceLight, borderRadius: 3, overflow: 'hidden' },
  barFill:     { height: '100%', borderRadius: 3 },
  barCount:    { color: COLORS.textSecondary, fontSize: 12, width: 28, textAlign: 'right' },

  infoText:    { color: COLORS.textSecondary, fontSize: 13, lineHeight: 20 },
});
