'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Sparkles, 
  RotateCcw, 
  History, 
  Maximize2, 
  Minimize2, 
  Volume2, 
  VolumeX,
  UserCheck, 
  AlertCircle,
  Copy,
  Check,
  Award,
  Layers
} from 'lucide-react';
import confetti from 'canvas-confetti';
import { Student, DrawMode, DrawRecord } from '@/lib/types';
import { soundFx } from '@/lib/sound';

interface PickerViewProps {
  students: Student[];
  onNavigateToRoster: () => void;
}

export const PickerView: React.FC<PickerViewProps> = ({
  students,
  onNavigateToRoster
}) => {
  const [drawMode, setDrawMode] = useState<DrawMode>('unique'); // default: 不重複抽取
  const [drawnStudentIds, setDrawnStudentIds] = useState<string[]>([]);
  const [history, setHistory] = useState<DrawRecord[]>([]);

  // Current display candidate while rolling or revealed winner
  const [currentCandidate, setCurrentCandidate] = useState<Student | null>(null);
  const [winner, setWinner] = useState<Student | null>(null);
  const [isRolling, setIsRolling] = useState(false);
  const [copiedHistory, setCopiedHistory] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const rollTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Available students depending on mode
  const availableStudents = drawMode === 'unique'
    ? students.filter(s => !drawnStudentIds.includes(s.id))
    : students;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rollTimerRef.current) clearTimeout(rollTimerRef.current);
    };
  }, []);

  // Handle Fullscreen toggle
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Main Draw Mechanism with dynamic deceleration
  const startDraw = useCallback(() => {
    if (students.length === 0) return;
    if (drawMode === 'unique' && availableStudents.length === 0) return;

    setIsRolling(true);
    setWinner(null);

    // Pick final winner in advance
    const pool = drawMode === 'unique' ? availableStudents : students;
    const randomIndex = Math.floor(Math.random() * pool.length);
    const selectedWinner = pool[randomIndex];

    // Build rolling pool from all students for exciting visual flurry
    const visualPool = students.length > 1 ? students : pool;

    let elapsedTicks = 0;
    const totalTicks = 32; // number of rapid switches
    let currentDelay = 45; // starts very fast (45ms)

    const step = () => {
      elapsedTicks++;
      
      // Random candidate for visual rolling
      const tempIndex = Math.floor(Math.random() * visualPool.length);
      setCurrentCandidate(visualPool[tempIndex]);

      // Sound effect: tick with slight pitch variation
      const pitch = 0.85 + (elapsedTicks / totalTicks) * 0.45;
      soundFx.playTick(pitch);

      if (elapsedTicks >= totalTicks) {
        // Stop on the chosen winner!
        setCurrentCandidate(selectedWinner);
        setWinner(selectedWinner);
        setIsRolling(false);

        // Sound: Fanfare celebration!
        soundFx.playFanfare();

        // Confetti explosion
        try {
          confetti({
            particleCount: 85,
            spread: 75,
            origin: { y: 0.6 },
            colors: ['#4f46e5', '#06b6d4', '#10b981', '#f59e0b', '#ec4899']
          });
        } catch {}

        // Update record
        const record: DrawRecord = {
          id: `rec-${Date.now()}`,
          student: selectedWinner,
          timestamp: Date.now()
        };
        setHistory(prev => [record, ...prev]);

        // If unique mode, add to drawn list
        if (drawMode === 'unique') {
          setDrawnStudentIds(prev => [...prev, selectedWinner.id]);
        }
      } else {
        // Deceleration curve: speed slows down during the last 40% of ticks
        if (elapsedTicks > totalTicks * 0.6) {
          currentDelay += 28;
        } else if (elapsedTicks > totalTicks * 0.3) {
          currentDelay += 8;
        }
        rollTimerRef.current = setTimeout(step, currentDelay);
      }
    };

    step();
  }, [students, drawMode, availableStudents]);

  // Keyboard shortcut: Spacebar triggers draw if not rolling
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && (e.target as HTMLElement).tagName !== 'INPUT' && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (!isRolling && availableStudents.length > 0) {
          startDraw();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRolling, availableStudents, startDraw]);

  // Reset unique pool
  const handleResetPool = () => {
    soundFx.playSoftClick();
    setDrawnStudentIds([]);
    setWinner(null);
    setCurrentCandidate(null);
  };

  // Clear history
  const handleClearHistory = () => {
    soundFx.playSoftClick();
    setHistory([]);
  };

  // Copy history text
  const handleCopyHistory = () => {
    if (history.length === 0) return;
    const text = history
      .map((h, i) => `${history.length - i}. ${h.student.seatNo ? `(${h.student.seatNo}號) ` : ''}${h.student.name} - ${new Date(h.timestamp).toLocaleTimeString()}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopiedHistory(true);
    soundFx.playSoftClick();
    setTimeout(() => setCopiedHistory(false), 2000);
  };

  if (students.length === 0) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <div className="w-16 h-16 rounded-2xl bg-indigo-100 text-indigo-600 flex items-center justify-center mx-auto mb-4">
          <AlertCircle className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-slate-800 mb-2">尚未匯入學生名單</h2>
        <p className="text-sm text-slate-500 mb-6">
          抽籤前需要至少一位學生名單，請先前往名單管理匯入 CSV 或貼上姓名。
        </p>
        <button
          id="btn-goto-roster-empty"
          onClick={onNavigateToRoster}
          className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold shadow-xs transition"
        >
          前往名單管理
        </button>
      </div>
    );
  }

  const isPoolExhausted = drawMode === 'unique' && availableStudents.length === 0;

  return (
    <div 
      ref={containerRef}
      className={`max-w-5xl mx-auto px-4 py-6 ${isFullscreen ? 'bg-slate-900 text-white min-h-screen flex flex-col justify-center' : ''}`}
    >
      {/* Mode Controls & Header Bar */}
      <div className={`flex flex-col sm:flex-row items-center justify-between gap-4 mb-6 ${isFullscreen ? 'hidden' : ''}`}>
        {/* Draw Mode Switch: Duplicate vs Non-duplicate */}
        <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-xl border border-slate-200">
          <button
            id="btn-mode-unique"
            onClick={() => {
              soundFx.playSoftClick();
              setDrawMode('unique');
            }}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              drawMode === 'unique'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <UserCheck className="w-3.5 h-3.5" />
            不重複抽取
            <span className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] bg-indigo-50 text-indigo-700 border border-indigo-200">
              剩 {availableStudents.length}/{students.length}
            </span>
          </button>

          <button
            id="btn-mode-repeat"
            onClick={() => {
              soundFx.playSoftClick();
              setDrawMode('repeat');
            }}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              drawMode === 'repeat'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            可重複抽取
            <span className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] bg-slate-200 text-slate-700">
              全體 {students.length}
            </span>
          </button>
        </div>

        {/* Right tools: Fullscreen projector toggle */}
        <div className="flex items-center gap-2">
          {drawMode === 'unique' && drawnStudentIds.length > 0 && (
            <button
              id="btn-reset-pool"
              onClick={handleResetPool}
              title="重置抽籤池，讓所有人重新可抽"
              className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-100 transition flex items-center gap-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              重置抽籤池
            </button>
          )}

          <button
            id="btn-toggle-fullscreen"
            onClick={toggleFullscreen}
            title={isFullscreen ? '離開全螢幕' : '全螢幕投影模式'}
            className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-100 transition flex items-center gap-1"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            <span>{isFullscreen ? '縮小' : '全螢幕投影'}</span>
          </button>
        </div>
      </div>

      {/* Main Drawing Stage Card */}
      <div 
        className={`relative overflow-hidden rounded-3xl transition-all duration-300 ${
          isFullscreen 
            ? 'bg-slate-800/80 border border-slate-700 p-12 max-w-4xl mx-auto w-full shadow-2xl text-center' 
            : 'bg-white border border-slate-200 shadow-lg p-8 sm:p-12 text-center'
        }`}
      >
        {/* Fullscreen exit button in upper right */}
        {isFullscreen && (
          <button
            onClick={toggleFullscreen}
            className="absolute top-6 right-6 p-2 rounded-xl bg-slate-700 hover:bg-slate-600 text-slate-200 transition"
            title="退出全螢幕"
          >
            <Minimize2 className="w-5 h-5" />
          </button>
        )}

        {/* Status Badge */}
        <div className="flex justify-center mb-4">
          <span className={`inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
            isRolling 
              ? 'bg-amber-100 text-amber-800 animate-pulse border border-amber-300' 
              : winner 
              ? 'bg-emerald-100 text-emerald-800 border border-emerald-300' 
              : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
          }`}>
            <Sparkles className="w-3.5 h-3.5" />
            {isRolling ? '幸運之輪轉動中...' : winner ? '🎉 恭喜中籤！' : '等待抽籤'}
          </span>
        </div>

        {/* Huge Name Display Screen */}
        <div className="my-6 min-h-[160px] sm:min-h-[200px] flex flex-col items-center justify-center">
          {isPoolExhausted ? (
            <div className="space-y-3">
              <div className="text-3xl sm:text-4xl font-extrabold text-amber-600">
                🎊 全班所有學生已全部抽過一輪！
              </div>
              <p className="text-sm text-slate-500">
                目前不重複模式下抽籤池已空，可點擊下方按鈕重置抽籤池。
              </p>
              <button
                id="btn-exhausted-reset"
                onClick={handleResetPool}
                className="mt-4 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-sm shadow-md transition"
              >
                重置抽籤池重新開始
              </button>
            </div>
          ) : currentCandidate || winner ? (
            <div className={`transition-transform duration-150 ${winner ? 'scale-105' : 'scale-100'}`}>
              {/* Seat number if available */}
              {(winner?.seatNo || currentCandidate?.seatNo) && (
                <div className="text-sm sm:text-lg font-bold text-indigo-600/80 mb-1 tracking-wider">
                  座號 {(winner || currentCandidate)?.seatNo} 號
                </div>
              )}
              {/* Student Name */}
              <div 
                id="winner-student-name"
                className={`text-5xl sm:text-7xl lg:text-8xl font-black tracking-tight leading-tight transition-all ${
                  isRolling 
                    ? 'text-slate-700 blur-[0.5px]' 
                    : winner 
                    ? (isFullscreen ? 'text-emerald-400 drop-shadow-[0_0_20px_rgba(52,211,153,0.4)]' : 'text-indigo-600 drop-shadow-sm') 
                    : (isFullscreen ? 'text-white' : 'text-slate-800')
                }`}
              >
                {(winner || currentCandidate)?.name}
              </div>
            </div>
          ) : (
            <div className="text-4xl sm:text-6xl font-bold text-slate-300 select-none">
              ？ ？ ？
            </div>
          )}
        </div>

        {/* Main Action Trigger */}
        {!isPoolExhausted && (
          <div className="mt-4 flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              id="btn-start-draw"
              disabled={isRolling}
              onClick={startDraw}
              className={`w-full sm:w-auto min-w-[220px] px-8 py-4 rounded-2xl text-lg font-extrabold shadow-lg transition-all flex items-center justify-center gap-2.5 ${
                isRolling
                  ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white shadow-indigo-200 hover:shadow-indigo-300'
              }`}
            >
              <Sparkles className="w-5 h-5" />
              <span>{isRolling ? '抽籤進行中...' : winner ? '再抽下一位' : '開始隨機抽籤'}</span>
            </button>
          </div>
        )}

        {/* Keyboard hint */}
        <p className="text-xs text-slate-400 mt-4">
          提示：也可以直接按下鍵盤 <kbd className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-300 text-slate-600 font-mono text-[11px]">Space 空白鍵</kbd> 進行抽籤
        </p>
      </div>

      {/* Pools & History Section (shown when not fullscreen) */}
      {!isFullscreen && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-8">
          {/* Left: Pool Status (Remaining vs Drawn) */}
          <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-600" />
                抽籤池狀態檢視
              </h3>
              <span className="text-xs text-slate-500 font-medium">
                {drawMode === 'unique' ? '不重複模式' : '可重複模式'}
              </span>
            </div>

            {drawMode === 'unique' ? (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-2">
                    <span>剩餘待抽學生 ({availableStudents.length})</span>
                    <span className="text-indigo-600 font-bold">
                      {Math.round((availableStudents.length / students.length) * 100)}%
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-2 bg-slate-50 rounded-xl border border-slate-100">
                    {availableStudents.length === 0 ? (
                      <span className="text-xs text-slate-400 italic">無剩餘學生</span>
                    ) : (
                      availableStudents.map(s => (
                        <span 
                          key={s.id}
                          className="px-2 py-1 rounded-md bg-white border border-slate-200 text-xs font-medium text-slate-800 shadow-2xs"
                        >
                          {s.seatNo ? `${s.seatNo}.` : ''}{s.name}
                        </span>
                      ))
                    )}
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-500 mb-2">
                    <span>已抽中名單 ({drawnStudentIds.length})</span>
                    {drawnStudentIds.length > 0 && (
                      <button
                        onClick={handleResetPool}
                        className="text-[11px] text-rose-600 hover:underline"
                      >
                        重置此名單
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto p-2 bg-slate-50 rounded-xl border border-slate-100">
                    {drawnStudentIds.length === 0 ? (
                      <span className="text-xs text-slate-400 italic">尚未有學生中籤</span>
                    ) : (
                      drawnStudentIds.map(id => {
                        const s = students.find(item => item.id === id);
                        if (!s) return null;
                        return (
                          <span 
                            key={id}
                            className="px-2 py-1 rounded-md bg-emerald-50 border border-emerald-200 text-xs font-medium text-emerald-800 flex items-center gap-1 line-through opacity-75"
                          >
                            <Check className="w-3 h-3 text-emerald-600" />
                            {s.seatNo ? `${s.seatNo}.` : ''}{s.name}
                          </span>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70 text-xs text-slate-600 leading-relaxed">
                目前處於「可重複抽取」模式。每次抽籤全班 {students.length} 位學生皆具備均等的中籤機率，抽中的學生不會從抽籤池中排除。
              </div>
            )}
          </div>

          {/* Right: Draw History */}
          <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 p-5 shadow-xs flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <History className="w-4 h-4 text-indigo-600" />
                抽籤歷史紀錄
              </h3>
              <div className="flex items-center gap-2">
                {history.length > 0 && (
                  <>
                    <button
                      id="btn-copy-history"
                      onClick={handleCopyHistory}
                      className="text-xs text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1"
                    >
                      {copiedHistory ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                      {copiedHistory ? '已複製！' : '複製紀錄'}
                    </button>
                    <span className="text-slate-300">|</span>
                    <button
                      id="btn-clear-history"
                      onClick={handleClearHistory}
                      className="text-xs text-slate-400 hover:text-rose-600 font-medium"
                    >
                      清空
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto max-h-56 divide-y divide-slate-100">
              {history.length === 0 ? (
                <div className="py-10 text-center text-xs text-slate-400">
                  尚無抽籤紀錄，點擊上方按鈕開始
                </div>
              ) : (
                history.map((record, index) => (
                  <div key={record.id} className="py-2.5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 text-[11px] font-bold flex items-center justify-center">
                        {history.length - index}
                      </span>
                      <span className="text-xs font-bold text-slate-800">
                        {record.student.name}
                      </span>
                      {record.student.seatNo && (
                        <span className="text-[11px] text-slate-400">
                          (座號 {record.student.seatNo})
                        </span>
                      )}
                    </div>
                    <span className="text-[11px] text-slate-400">
                      {new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
