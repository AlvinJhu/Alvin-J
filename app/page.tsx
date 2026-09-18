'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Student } from '@/lib/types';
import { DEFAULT_STUDENTS } from '@/lib/roster-parser';
import { Navbar } from '@/components/Navbar';
import { PickerView } from '@/components/PickerView';
import { GroupingView } from '@/components/GroupingView';
import { RosterManager } from '@/components/RosterManager';
import { soundFx } from '@/lib/sound';

export default function Home() {
  const [students, setStudents] = useState<Student[]>(() => {
    if (typeof window === 'undefined') return DEFAULT_STUDENTS;
    try {
      const saved = localStorage.getItem('classroom_roster_students');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch {}
    return DEFAULT_STUDENTS;
  });

  const [activeTab, setActiveTab] = useState<'picker' | 'grouping' | 'roster'>('picker');
  const [isMuted, setIsMuted] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      const saved = localStorage.getItem('classroom_sound_muted');
      const muted = saved === 'true';
      soundFx.isMuted = muted;
      return muted;
    } catch {
      return false;
    }
  });

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('classroom_roster_students', JSON.stringify(students));
    } catch {}
  }, [students]);

  useEffect(() => {
    try {
      localStorage.setItem('classroom_sound_muted', isMuted ? 'true' : 'false');
    } catch {}
  }, [isMuted]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 selection:bg-indigo-500 selection:text-white">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        studentCount={students.length}
        isMuted={isMuted}
        setIsMuted={setIsMuted}
      />

      {/* Main Content Area */}
      <main className="flex-1 pb-16">
        <AnimatePresence mode="wait">
          {activeTab === 'picker' && (
            <motion.div
              key="picker-tab"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18 }}
            >
              <PickerView
                students={students}
                onNavigateToRoster={() => setActiveTab('roster')}
              />
            </motion.div>
          )}

          {activeTab === 'grouping' && (
            <motion.div
              key="grouping-tab"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18 }}
            >
              <GroupingView
                students={students}
                onNavigateToRoster={() => setActiveTab('roster')}
              />
            </motion.div>
          )}

          {activeTab === 'roster' && (
            <motion.div
              key="roster-tab"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18 }}
            >
              <RosterManager
                students={students}
                setStudents={setStudents}
                onNavigateToPicker={() => setActiveTab('picker')}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Subtle Teacher Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-4 text-center text-xs text-slate-400 print:hidden">
        <p>班級隨機抽籤與分組系統 · 支援 CSV 匯入、動畫音效抽籤與公平隨機分組</p>
      </footer>
    </div>
  );
}
