import React, { useState, useCallback } from 'react';
import {
  View, Text, FlatList, StyleSheet, RefreshControl,
  ActivityIndicator, StatusBar, TouchableOpacity,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, IRI_COLORS, IRI_LABELS, IriCategory } from '../config/env';
import { observationService } from '../services/observationService';

function iriToCategory(score: number): IriCategory {
  if (score < 1.5)  return 'green';
  if (score < 2.5) return 'yellow';
  return 'orange';
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1)   return 'just now';
  if (mins < 60)  return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)   return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

interface ObsItem {
  _id:       string;
  latitude:  number;
  longitude: number;
  iriScore:  number;
  hasPothole: boolean;
  recordedAt: string;
}

const Item = ({ item }: { item: ObsItem }) => {
  const cat   = iriToCategory(item.iriScore);
  const color = IRI_COLORS[cat];
  return (
    <View style={styles.item}>
      <View style={[styles.itemBar, { backgroundColor: color }]} />
      <View style={styles.itemBody}>
        <View style={styles.itemTop}>
          <Text style={styles.itemLabel}>{IRI_LABELS[cat]}</Text>
          {item.hasPothole && <View style={styles.potholePill}><Text style={styles.potholeText}>Pothole</Text></View>}
        </View>
        <Text style={styles.itemCoords}>
          {item.latitude.toFixed(5)}, {item.longitude.toFixed(5)}
        </Text>
        <Text style={styles.itemTime}>{timeAgo(item.recordedAt)}</Text>
      </View>
      <View style={styles.itemScore}>
        <Text style={[styles.iriValue, { color }]}>{item.iriScore.toFixed(1)}</Text>
        <Text style={styles.iriUnit}>IRI</Text>
      </View>
    </View>
  );
};

export const HistoryScreen = () => {
  const [data,       setData]       = useState<ObsItem[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const history = await observationService.getHistory();
      setData(history as ObsItem[]);
    } catch (e) {
      console.warn(e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const onRefresh = () => { setRefreshing(true); load(); };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />
      <View style={styles.header}>
        <Text style={styles.headerTitle}>My Reports</Text>
        <Text style={styles.headerSub}>{data.length} observations</Text>
      </View>
      {data.length === 0 ? (
        <View style={styles.center}>
          <Text style={styles.emptyIcon}>🛣️</Text>
          <Text style={styles.emptyTitle}>No reports yet</Text>
          <Text style={styles.emptySub}>Start monitoring to contribute data</Text>
        </View>
      ) : (
        <FlatList
          data={data}
          keyExtractor={i => i._id}
          renderItem={({ item }) => <Item item={item} />}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={COLORS.primary}
            />
          }
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: COLORS.bg },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: COLORS.bg },

  header: {
    paddingHorizontal: 20,
    paddingTop:        60,
    paddingBottom:     20,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  headerTitle: { color: COLORS.text,          fontSize: 22, fontWeight: '700' },
  headerSub:   { color: COLORS.textSecondary,  fontSize: 13, marginTop: 2 },

  list: { padding: 16, gap: 10 },

  item: {
    flexDirection:   'row',
    backgroundColor: COLORS.surface,
    borderRadius:    14,
    overflow:        'hidden',
    borderWidth:     1,
    borderColor:     COLORS.border,
  },
  itemBar:    { width: 4 },
  itemBody:   { flex: 1, padding: 14, gap: 4 },
  itemTop:    { flexDirection: 'row', alignItems: 'center', gap: 8 },
  itemLabel:  { color: COLORS.text, fontWeight: '600', fontSize: 15 },
  itemCoords: { color: COLORS.textMuted, fontSize: 11, fontFamily: 'monospace' },
  itemTime:   { color: COLORS.textSecondary, fontSize: 12 },

  potholePill: {
    backgroundColor: '#F4433622',
    borderRadius:    6,
    paddingHorizontal: 8,
    paddingVertical:   2,
    borderWidth:     1,
    borderColor:     '#F44336',
  },
  potholeText: { color: '#F44336', fontSize: 10, fontWeight: '700' },

  itemScore:  { padding: 14, justifyContent: 'center', alignItems: 'center' },
  iriValue:   { fontSize: 20, fontWeight: '700' },
  iriUnit:    { color: COLORS.textMuted, fontSize: 10, marginTop: 2 },

  emptyIcon:  { fontSize: 48, marginBottom: 12 },
  emptyTitle: { color: COLORS.text,          fontSize: 18, fontWeight: '600' },
  emptySub:   { color: COLORS.textSecondary,  fontSize: 14, marginTop: 6 },
});
