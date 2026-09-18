'use client';

import React from 'react';
import { 
  Dices, 
  Users, 
  ClipboardList, 
  Volume2, 
  VolumeX, 
  GraduationCap,
  Sparkles
} from 'lucide-react';
import { soundFx } from '@/lib/sound';

interface NavbarProps {
  activeTab: 'picker' | 'grouping' | 'roster';
  setActiveTab: (tab: 'picker' | 'grouping' | 'roster') => void;
  studentCount: number;
  isMuted: boolean;
  setIsMuted: (muted: boolean) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  studentCount,
  isMuted,
  setIsMuted
}) => {
  const toggleSound = () => {
    const nextMuted = !isMuted;
    soundFx.isMuted = nextMuted;
    setIsMuted(nextMuted);
    if (!nextMuted) {
      soundFx.playSoftClick();
    }
  };

  const handleTabClick = (tab: 'picker' | 'grouping' | 'roster') => {
    soundFx.playSoftClick();
    setActiveTab(tab);
  };

  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-sm border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-xs">
              <GraduationCap className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-slate-900 tracking-tight">
                  班級隨機抽籤與分組系統
                </h1>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
                  教師助手
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                名單匯入 · 動態音效抽籤 · 視覺化自動分組
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200">
            <button
              id="tab-btn-picker"
              onClick={() => handleTabClick('picker')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'picker'
                  ? 'bg-white text-indigo-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Dices className="w-4 h-4 text-indigo-600" />
              <span>隨機抽籤</span>
            </button>

            <button
              id="tab-btn-grouping"
              onClick={() => handleTabClick('grouping')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'grouping'
                  ? 'bg-white text-emerald-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Users className="w-4 h-4 text-emerald-600" />
              <span>自動分組</span>
            </button>

            <button
              id="tab-btn-roster"
              onClick={() => handleTabClick('roster')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'roster'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <ClipboardList className="w-4 h-4 text-slate-600" />
              <span>名單管理</span>
              <span className="ml-1 px-1.5 py-0.2 rounded-full text-xs bg-slate-200 text-slate-700 font-semibold">
                {studentCount}
              </span>
            </button>
          </nav>

          {/* Right Tools */}
          <div className="flex items-center gap-2">
            <button
              id="toggle-sound-btn"
              onClick={toggleSound}
              title={isMuted ? '開啟音效' : '靜音'}
              aria-label={isMuted ? '開啟音效' : '靜音'}
              className={`p-2 rounded-lg border transition-colors ${
                isMuted
                  ? 'bg-slate-100 border-slate-200 text-slate-400 hover:text-slate-700'
                  : 'bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100'
              }`}
            >
              {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
