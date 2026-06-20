import React, { useEffect, useState } from 'react';
import {
  View, ActivityIndicator, StatusBar, Text, StyleSheet,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import { LoginScreen }   from './src/screens/LoginScreen';
import { MapScreen }     from './src/screens/MapScreen';
import { HistoryScreen } from './src/screens/HistoryScreen';
import { StatsScreen }   from './src/screens/StatsScreen';
import { authService }   from './src/services/authService';
import { COLORS }        from './src/config/env';

// ─── Navigators ───────────────────────────────────────────────────────────────
const Stack = createStackNavigator();
const Tab   = createBottomTabNavigator();

// ─── Tab icons (emoji — no icon library needed) ───────────────────────────────
const TAB_ICONS: Record<string, string> = {
  Map:     '🗺',
  History: '📋',
  Stats:   '📊',
};

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused }) => (
          <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.5 }}>
            {TAB_ICONS[route.name]}
          </Text>
        ),
        tabBarLabel: ({ focused, children }) => (
          <Text style={{
            fontSize:   10,
            fontWeight: focused ? '700' : '400',
            color:      focused ? COLORS.primary : COLORS.textMuted,
            marginBottom: 4,
          }}>
            {children}
          </Text>
        ),
        tabBarStyle: {
          backgroundColor:   COLORS.surface,
          borderTopColor:    COLORS.border,
          borderTopWidth:    1,
          height:            60,
          paddingTop:        6,
        },
        tabBarActiveTintColor:   COLORS.primary,
        tabBarInactiveTintColor: COLORS.textMuted,
      })}
    >
      <Tab.Screen name="Map"     component={MapScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Stats"   component={StatsScreen} />
    </Tab.Navigator>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [checking, setChecking] = useState(true);
  const [authed,   setAuthed]   = useState(false);

  useEffect(() => {
    authService.getToken().then(token => {
      setAuthed(!!token);
      setChecking(false);
    });
  }, []);

  if (checking) {
    return (
      <View style={styles.splash}>
        <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />
        <Text style={styles.splashIcon}>🛣️</Text>
        <Text style={styles.splashTitle}>Road Monitor</Text>
        <ActivityIndicator color={COLORS.primary} style={{ marginTop: 24 }} />
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
        <Stack.Navigator
          initialRouteName={authed ? 'Main' : 'Login'}
          screenOptions={{ headerShown: false }}
        >
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="Main"  component={MainTabs} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex:            1,
    backgroundColor: COLORS.bg,
    justifyContent:  'center',
    alignItems:      'center',
  },
  splashIcon:  { fontSize: 56, marginBottom: 12 },
  splashTitle: { color: COLORS.text, fontSize: 24, fontWeight: '700' },
});
