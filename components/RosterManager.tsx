'use client';

import React, { useState, useRef } from 'react';
import { 
  Upload, 
  FileText, 
  Trash2, 
  Plus, 
  RotateCcw, 
  Download, 
  Check, 
  Search, 
  AlertCircle,
  Sparkles,
  FileSpreadsheet
} from 'lucide-react';
import { Student } from '@/lib/types';
import { parseRosterText, DEFAULT_STUDENTS } from '@/lib/roster-parser';
import { soundFx } from '@/lib/sound';

interface RosterManagerProps {
  students: Student[];
  setStudents: (students: Student[]) => void;
  onNavigateToPicker?: () => void;
}

export const RosterManager: React.FC<RosterManagerProps> = ({
  students,
  setStudents,
  onNavigateToPicker
}) => {
  const [activeInputMode, setActiveInputMode] = useState<'paste' | 'csv'>('paste');
  const [pasteText, setPasteText] = useState('');
  const [newStudentName, setNewStudentName] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'info' | 'error'; text: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showFeedback = (text: string, type: 'success' | 'info' | 'error' = 'success') => {
    setStatusMessage({ type, text });
    setTimeout(() => {
      setStatusMessage(null);
    }, 4000);
  };

  // Process text paste
  const handleApplyPastedText = () => {
    if (!pasteText.trim()) {
      showFeedback('請先貼上或輸入學生姓名', 'error');
      return;
    }
    const parsed = parseRosterText(pasteText);
    if (parsed.length === 0) {
      showFeedback('未能識別到有效學生姓名，請檢查格式', 'error');
      return;
    }
    setStudents(parsed);
    soundFx.playSoftClick();
    showFeedback(`成功匯入 ${parsed.length} 位學生名單！`, 'success');
  };

  // Process CSV File
  const handleFile = (file: File) => {
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
      showFeedback('請上傳 .csv 或 .txt 格式的名單檔案', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (!content) {
        showFeedback('讀取檔案失敗或內容為空', 'error');
        return;
      }
      const parsed = parseRosterText(content);
      if (parsed.length === 0) {
        showFeedback('檔案內未找到有效學生名單，請確認內容格式', 'error');
        return;
      }
      setStudents(parsed);
      soundFx.playSoftClick();
      showFeedback(`已從 ${file.name} 成功匯入 ${parsed.length} 位學生名單！`, 'success');
    };
    reader.onerror = () => {
      showFeedback('讀取檔案時發生錯誤', 'error');
    };
    reader.readAsText(file, 'utf-8');
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  // Add individual student
  const handleAddSingleStudent = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newStudentName.trim();
    if (!trimmed) return;

    const nextSeat = students.length > 0 ? Math.max(...students.map(s => s.seatNo || 0)) + 1 : 1;
    const newStudent: Student = {
      id: `std-${Date.now()}`,
      seatNo: nextSeat,
      name: trimmed
    };

    setStudents([...students, newStudent]);
    setNewStudentName('');
    soundFx.playSoftClick();
    showFeedback(`已新增學生：${trimmed}`, 'success');
  };

  // Delete individual student
  const handleDeleteStudent = (id: string, name: string) => {
    setStudents(students.filter(s => s.id !== id));
    soundFx.playSoftClick();
    showFeedback(`已移除：${name}`, 'info');
  };

  // Clear all
  const handleClearAll = () => {
    if (window.confirm('確定要清空目前所有學生名單嗎？')) {
      setStudents([]);
      soundFx.playSoftClick();
      showFeedback('名單已清空', 'info');
    }
  };

  // Load sample roster
  const handleLoadSample = () => {
    setStudents(DEFAULT_STUDENTS);
    soundFx.playSoftClick();
    showFeedback('已成功載入 30 人示範名單！', 'success');
  };

  // Download sample CSV
  const handleDownloadSampleCsv = () => {
    const csvContent = '\uFEFF座號,姓名\n' + DEFAULT_STUDENTS.map(s => `${s.seatNo},${s.name}`).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', '學生名單範例.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    soundFx.playSoftClick();
  };

  // Export current roster to CSV
  const handleExportCurrentRoster = () => {
    if (students.length === 0) {
      showFeedback('目前名單為空，無法匯出', 'error');
      return;
    }
    const csvContent = '\uFEFF座號,姓名\n' + students.map(s => `${s.seatNo || ''},${s.name}`).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `班級學生名單_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    soundFx.playSoftClick();
    showFeedback('名單匯出完成！', 'success');
  };

  // Filtered students for display
  const filteredStudents = students.filter(s => 
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    (s.seatNo && s.seatNo.toString().includes(searchQuery))
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Top Banner Status Message */}
      {statusMessage && (
        <div 
          id="roster-status-alert"
          className={`mb-6 p-4 rounded-xl flex items-center gap-3 transition-all ${
            statusMessage.type === 'success' 
              ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' 
              : statusMessage.type === 'error'
              ? 'bg-rose-50 text-rose-800 border border-rose-200'
              : 'bg-indigo-50 text-indigo-800 border border-indigo-200'
          }`}
        >
          {statusMessage.type === 'success' ? (
            <Check className="w-5 h-5 text-emerald-600 shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 shrink-0" />
          )}
          <span className="text-sm font-medium">{statusMessage.text}</span>
        </div>
      )}

      {/* Main Grid: Input Setup on Left, Roster List on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Import / Input methods */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
            <div className="p-5 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900">
                  匯入名單來源
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  支援上傳 CSV 檔案或直接貼上學生姓名
                </p>
              </div>

              {/* Mode switch */}
              <div className="flex bg-slate-200/70 p-1 rounded-lg">
                <button
                  id="tab-input-paste"
                  onClick={() => {
                    soundFx.playSoftClick();
                    setActiveInputMode('paste');
                  }}
                  className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                    activeInputMode === 'paste'
                      ? 'bg-white text-indigo-700 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5 inline mr-1" />
                  貼上名單
                </button>
                <button
                  id="tab-input-csv"
                  onClick={() => {
                    soundFx.playSoftClick();
                    setActiveInputMode('csv');
                  }}
                  className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                    activeInputMode === 'csv'
                      ? 'bg-white text-indigo-700 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5 inline mr-1" />
                  上傳 CSV
                </button>
              </div>
            </div>

            <div className="p-6">
              {activeInputMode === 'paste' ? (
                <div className="space-y-4">
                  <div>
                    <label htmlFor="paste-textarea" className="block text-xs font-semibold text-slate-700 mb-1.5">
                      貼上文字名單（每行一位、或用逗號/空格隔開）
                    </label>
                    <textarea
                      id="paste-textarea"
                      rows={8}
                      value={pasteText}
                      onChange={(e) => setPasteText(e.target.value)}
                      placeholder="例如：&#10;陳小明&#10;林美美&#10;張大千&#10;或：&#10;1, 王品涵&#10;2, 李冠宇"
                      className="w-full px-3.5 py-2.5 rounded-xl border border-slate-300 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 font-mono transition"
                    />
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <button
                      id="btn-apply-paste"
                      onClick={handleApplyPastedText}
                      className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white text-sm font-semibold shadow-xs hover:shadow transition flex items-center gap-2"
                    >
                      <Check className="w-4 h-4" />
                      匯入文字名單
                    </button>

                    <button
                      id="btn-load-sample"
                      onClick={handleLoadSample}
                      className="text-xs text-indigo-600 hover:text-indigo-800 hover:underline flex items-center gap-1.5 font-medium"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      帶入 30 人示範名單
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-5">
                  <div
                    id="csv-drop-zone"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                      isDragging 
                        ? 'border-indigo-500 bg-indigo-50/50' 
                        : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'
                    }`}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv, .txt"
                      className="hidden"
                      onChange={handleFileInputChange}
                    />
                    <div className="w-12 h-12 rounded-full bg-indigo-100 text-indigo-600 mx-auto flex items-center justify-center mb-3">
                      <FileSpreadsheet className="w-6 h-6" />
                    </div>
                    <p className="text-sm font-semibold text-slate-800">
                      拖曳 CSV 檔案到此處，或 <span className="text-indigo-600 underline">點擊瀏覽檔案</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      支援標準 .csv 或 .txt（第一欄為座號或姓名皆可自動識別）
                    </p>
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-500 pt-1">
                    <button
                      id="btn-download-csv-sample"
                      onClick={handleDownloadSampleCsv}
                      className="text-indigo-600 hover:text-indigo-800 hover:underline flex items-center gap-1 font-medium"
                    >
                      <Download className="w-3.5 h-3.5" />
                      下載 CSV 格式範本
                    </button>

                    <button
                      id="btn-load-sample-csv-tab"
                      onClick={handleLoadSample}
                      className="text-slate-600 hover:text-slate-900 hover:underline flex items-center gap-1 font-medium"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                      直接使用 30 人示範名單
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Quick single student add */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5">
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">
              手動新增單一學生
            </h3>
            <form onSubmit={handleAddSingleStudent} className="flex gap-2">
              <input
                id="input-single-student-name"
                type="text"
                value={newStudentName}
                onChange={(e) => setNewStudentName(e.target.value)}
                placeholder="輸入學生姓名（例如：王小明）"
                className="flex-1 px-3.5 py-2 rounded-xl border border-slate-300 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <button
                id="btn-add-single-student"
                type="submit"
                className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-sm font-semibold transition flex items-center gap-1.5 shrink-0"
              >
                <Plus className="w-4 h-4" />
                新增
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Active Roster Table & Quick Stats */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xs flex flex-col h-full min-h-[480px]">
            {/* Header with Search & Actions */}
            <div className="p-5 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-bold text-slate-900">
                    目前班級名單
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-100 text-indigo-800">
                    共 {students.length} 位
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  可用於隨機抽籤與自動分組功能
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  id="btn-export-csv"
                  onClick={handleExportCurrentRoster}
                  disabled={students.length === 0}
                  title="匯出名單為 CSV"
                  className="px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  匯出
                </button>

                <button
                  id="btn-clear-all"
                  onClick={handleClearAll}
                  disabled={students.length === 0}
                  title="清空名單"
                  className="px-2.5 py-1.5 rounded-lg border border-rose-200 text-xs font-medium text-rose-600 hover:bg-rose-50 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 transition"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  清空
                </button>
              </div>
            </div>

            {/* Search filter input */}
            <div className="px-5 pt-3 pb-2 border-b border-slate-100">
              <div className="relative">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  id="input-roster-search"
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="搜尋姓名或座號..."
                  className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-800 focus:outline-none focus:bg-white focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* List Content */}
            <div className="flex-1 overflow-y-auto max-h-[380px] p-4">
              {students.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-56 text-center text-slate-400">
                  <FileSpreadsheet className="w-12 h-12 stroke-[1.5] mb-2 text-slate-300" />
                  <p className="text-sm font-medium text-slate-600">目前名單尚無學生</p>
                  <p className="text-xs text-slate-400 mt-1">請在上傳 CSV 或貼上文字名單後開始</p>
                  <button
                    onClick={handleLoadSample}
                    className="mt-3 px-3.5 py-1.5 rounded-lg bg-indigo-50 text-indigo-600 text-xs font-semibold hover:bg-indigo-100 transition"
                  >
                    載入 30 人示範名單
                  </button>
                </div>
              ) : filteredStudents.length === 0 ? (
                <div className="text-center py-10 text-slate-400 text-xs">
                  找不到符合「{searchQuery}」的學生
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {filteredStudents.map((student, idx) => (
                    <div
                      key={student.id}
                      className="group flex items-center justify-between px-3 py-2 rounded-xl bg-slate-50 hover:bg-indigo-50/60 border border-slate-200/80 hover:border-indigo-200 transition"
                    >
                      <div className="flex items-center gap-2 overflow-hidden">
                        <span className="w-6 h-6 rounded-md bg-slate-200 text-slate-700 text-xs font-bold flex items-center justify-center shrink-0">
                          {student.seatNo || idx + 1}
                        </span>
                        <span className="text-sm font-semibold text-slate-800 truncate">
                          {student.name}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDeleteStudent(student.id, student.name)}
                        title="刪除此學生"
                        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-rose-600 p-1 transition"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Bottom Quick Call to Action */}
            {students.length > 0 && onNavigateToPicker && (
              <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  名單已備妥，可立即進行抽籤或分組
                </span>
                <button
                  id="btn-goto-picker"
                  onClick={() => {
                    soundFx.playSoftClick();
                    onNavigateToPicker();
                  }}
                  className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold shadow-xs transition flex items-center gap-1.5"
                >
                  開始隨機抽籤
                  <Sparkles className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
