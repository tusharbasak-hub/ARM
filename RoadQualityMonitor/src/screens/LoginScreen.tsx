import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform, ScrollView,
  StatusBar,
} from 'react-native';
import { authService } from '../services/authService';
import { COLORS } from '../config/env';

type Mode = 'login' | 'register';

export const LoginScreen = ({ navigation }: any) => {
  const [mode,     setMode]     = useState<Mode>('login');
  const [name,     setName]     = useState('');
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);

  const go = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert('Missing fields', 'Please enter email and password.');
      return;
    }
    if (mode === 'register' && !name.trim()) {
      Alert.alert('Missing fields', 'Please enter your name.');
      return;
    }
    setLoading(true);
    try {
      if (mode === 'login') {
        await authService.login(email.trim(), password);
      } else {
        await authService.register(email.trim(), password, name.trim());
      }
      navigation.replace('Main');
    } catch (e: any) {
      Alert.alert('Error', e.message ?? 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const goGuest = async () => {
    setLoading(true);
    try {
      await authService.guestLogin();
      navigation.replace('Main');
    } catch (e: any) {
      Alert.alert('Error', e.message ?? 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">

        {/* ── Logo / Heading ── */}
        <View style={styles.hero}>
          <View style={styles.iconWrap}>
            <Text style={styles.iconText}>🛣️</Text>
          </View>
          <Text style={styles.appName}>Road Monitor</Text>
          <Text style={styles.tagline}>Delhi NCR — Live Road Quality</Text>
        </View>

        {/* ── Card ── */}
        <View style={styles.card}>

          {/* Tab switcher */}
          <View style={styles.tabs}>
            {(['login', 'register'] as Mode[]).map(m => (
              <TouchableOpacity
                key={m}
                style={[styles.tab, mode === m && styles.tabActive]}
                onPress={() => setMode(m)}
              >
                <Text style={[styles.tabText, mode === m && styles.tabTextActive]}>
                  {m === 'login' ? 'Sign In' : 'Register'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Fields */}
          {mode === 'register' && (
            <View style={styles.inputWrap}>
              <Text style={styles.label}>Name</Text>
              <TextInput
                style={styles.input}
                placeholder="Your name"
                placeholderTextColor={COLORS.textMuted}
                value={name}
                onChangeText={setName}
              />
            </View>
          )}

          <View style={styles.inputWrap}>
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              placeholderTextColor={COLORS.textMuted}
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>

          <View style={styles.inputWrap}>
            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="••••••••"
              placeholderTextColor={COLORS.textMuted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>

          {/* Primary CTA */}
          <TouchableOpacity
            style={[styles.btn, styles.btnPrimary, loading && styles.btnDisabled]}
            onPress={go}
            disabled={loading}
            activeOpacity={0.8}
          >
            {loading
              ? <ActivityIndicator color="#000" />
              : <Text style={styles.btnPrimaryText}>
                  {mode === 'login' ? 'Sign In' : 'Create Account'}
                </Text>
            }
          </TouchableOpacity>

          {/* Divider */}
          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* Guest */}
          <TouchableOpacity
            style={[styles.btn, styles.btnGhost, loading && styles.btnDisabled]}
            onPress={goGuest}
            disabled={loading}
            activeOpacity={0.8}
          >
            <Text style={styles.btnGhostText}>👤  Continue as Guest</Text>
          </TouchableOpacity>

        </View>

        <Text style={styles.footer}>
          Your data helps improve roads for everyone.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: COLORS.bg },
  scroll: { flexGrow: 1, justifyContent: 'center', padding: 24 },

  hero:      { alignItems: 'center', marginBottom: 32 },
  iconWrap:  {
    width: 72, height: 72, borderRadius: 20,
    backgroundColor: COLORS.surfaceLight,
    justifyContent: 'center', alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1, borderColor: COLORS.border,
  },
  iconText:  { fontSize: 34 },
  appName:   { fontSize: 26, fontWeight: '700', color: COLORS.text, letterSpacing: 0.5 },
  tagline:   { fontSize: 13, color: COLORS.textSecondary, marginTop: 4 },

  card: {
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  tabs:          { flexDirection: 'row', marginBottom: 24, backgroundColor: COLORS.surfaceLight, borderRadius: 10, padding: 4 },
  tab:           { flex: 1, paddingVertical: 10, alignItems: 'center', borderRadius: 8 },
  tabActive:     { backgroundColor: COLORS.primary },
  tabText:       { color: COLORS.textSecondary, fontWeight: '600', fontSize: 14 },
  tabTextActive: { color: '#000' },

  inputWrap:  { marginBottom: 16 },
  label:      { color: COLORS.textSecondary, fontSize: 12, marginBottom: 6, fontWeight: '600', letterSpacing: 0.5 },
  input: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    padding: 14,
    color: COLORS.text,
    fontSize: 15,
  },

  btn:            { borderRadius: 12, padding: 16, alignItems: 'center', marginTop: 8 },
  btnPrimary:     { backgroundColor: COLORS.primary },
  btnPrimaryText: { color: '#000', fontWeight: '700', fontSize: 15 },
  btnGhost:       { backgroundColor: 'transparent', borderWidth: 1, borderColor: COLORS.border },
  btnGhostText:   { color: COLORS.textSecondary, fontWeight: '600', fontSize: 15 },
  btnDisabled:    { opacity: 0.5 },

  divider:      { flexDirection: 'row', alignItems: 'center', marginVertical: 16 },
  dividerLine:  { flex: 1, height: 1, backgroundColor: COLORS.border },
  dividerText:  { color: COLORS.textMuted, marginHorizontal: 12, fontSize: 12 },

  footer: { textAlign: 'center', color: COLORS.textMuted, fontSize: 12, marginTop: 24 },
});
