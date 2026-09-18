'use client';

import React, { useState, useEffect } from 'react';
import { 
  Users, 
  Shuffle, 
  Crown, 
  Copy, 
  Check, 
  Download, 
  ArrowRightLeft, 
  Sparkles,
  AlertCircle,
  Hash,
  Layers,
  Printer
} from 'lucide-react';
import { Student, StudentGroup, GroupMode, RemainderStrategy } from '@/lib/types';
import { soundFx } from '@/lib/sound';

interface GroupingViewProps {
  students: Student[];
  onNavigateToRoster: () => void;
}

const GROUP_PALETTES = [
  { header: 'bg-emerald-600 text-white', badge: 'bg-emerald-100 text-emerald-800', border: 'border-emerald-200' },
  { header: 'bg-indigo-600 text-white', badge: 'bg-indigo-100 text-indigo-800', border: 'border-indigo-200' },
  { header: 'bg-amber-600 text-white', badge: 'bg-amber-100 text-amber-800', border: 'border-amber-200' },
  { header: 'bg-sky-600 text-white', badge: 'bg-sky-100 text-sky-800', border: 'border-sky-200' },
  { header: 'bg-rose-600 text-white', badge: 'bg-rose-100 text-rose-800', border: 'border-rose-200' },
  { header: 'bg-purple-600 text-white', badge: 'bg-purple-100 text-purple-800', border: 'border-purple-200' },
  { header: 'bg-teal-600 text-white', badge: 'bg-teal-100 text-teal-800', border: 'border-teal-200' },
  { header: 'bg-orange-600 text-white', badge: 'bg-orange-100 text-orange-800', border: 'border-orange-200' },
  { header: 'bg-blue-600 text-white', badge: 'bg-blue-100 text-blue-800', border: 'border-blue-200' },
  { header: 'bg-fuchsia-600 text-white', badge: 'bg-fuchsia-100 text-fuchsia-800', border: 'border-fuchsia-200' },
];

export const GroupingView: React.FC<GroupingViewProps> = ({
  students,
  onNavigateToRoster
}) => {
  const [groupMode, setGroupMode] = useState<GroupMode>('bySize'); // 每組幾人 or 分幾組
  const [targetSize, setTargetSize] = useState<number>(4); // 每組預設 4 人
  const [targetGroupCount, setTargetGroupCount] = useState<number>(6); // 預設 6 組
  const [remainderStrategy, setRemainderStrategy] = useState<RemainderStrategy>('distribute');
  const [assignLeaders, setAssignLeaders] = useState<boolean>(true);

  const [groups, setGroups] = useState<StudentGroup[]>([]);
  const [copied, setCopied] = useState(false);
  const [selectedStudentToMove, setSelectedStudentToMove] = useState<{ student: Student; fromGroupId: string } | null>(null);

  // Perform random grouping using Fisher-Yates shuffle
  const generateGroups = () => {
    if (students.length === 0) return;

    soundFx.playSoftClick();

    // Shuffle copy of students
    const shuffled = [...students];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const n = shuffled.length;
    const resultGroups: StudentGroup[] = [];

    if (groupMode === 'bySize') {
      const size = Math.max(1, Math.min(targetSize, n));
      const quotient = Math.floor(n / size);
      const remainder = n % size;

      if (remainder === 0) {
        // Even split
        for (let i = 0; i < quotient; i++) {
          const members = shuffled.slice(i * size, (i + 1) * size);
          resultGroups.push({
            id: `grp-${i + 1}`,
            name: `第 ${i + 1} 組`,
            colorIndex: i % GROUP_PALETTES.length,
            members,
            leaderId: assignLeaders && members.length > 0 ? members[0].id : undefined
          });
        }
      } else if (remainderStrategy === 'distribute') {
        // Distribute remainder students into the first `remainder` groups
        const numGroups = Math.max(1, quotient);
        const buckets: Student[][] = Array.from({ length: numGroups }, () => []);
        
        shuffled.forEach((student, idx) => {
          const targetBucket = idx % numGroups;
          buckets[targetBucket].push(student);
        });

        buckets.forEach((members, i) => {
          resultGroups.push({
            id: `grp-${i + 1}`,
            name: `第 ${i + 1} 組`,
            colorIndex: i % GROUP_PALETTES.length,
            members,
            leaderId: assignLeaders && members.length > 0 ? members[0].id : undefined
          });
        });
      } else {
        // Remainder forms a separate group
        for (let i = 0; i < quotient; i++) {
          const members = shuffled.slice(i * size, (i + 1) * size);
          resultGroups.push({
            id: `grp-${i + 1}`,
            name: `第 ${i + 1} 組`,
            colorIndex: i % GROUP_PALETTES.length,
            members,
            leaderId: assignLeaders && members.length > 0 ? members[0].id : undefined
          });
        }
        if (remainder > 0) {
          const remainderMembers = shuffled.slice(quotient * size);
          resultGroups.push({
            id: `grp-${quotient + 1}`,
            name: `第 ${quotient + 1} 組`,
            colorIndex: quotient % GROUP_PALETTES.length,
            members: remainderMembers,
            leaderId: assignLeaders && remainderMembers.length > 0 ? remainderMembers[0].id : undefined
          });
        }
      }
    } else {
      // By total number of groups
      const numGroups = Math.max(1, Math.min(targetGroupCount, n));
      const buckets: Student[][] = Array.from({ length: numGroups }, () => []);

      shuffled.forEach((student, idx) => {
        const targetBucket = idx % numGroups;
        buckets[targetBucket].push(student);
      });

      buckets.forEach((members, i) => {
        resultGroups.push({
          id: `grp-${i + 1}`,
          name: `第 ${i + 1} 組`,
          colorIndex: i % GROUP_PALETTES.length,
          members,
          leaderId: assignLeaders && members.length > 0 ? members[0].id : undefined
        });
      });
    }

    setGroups(resultGroups);
  };

  // Re-assign team leaders randomly
  const handleReassignLeaders = () => {
    soundFx.playSoftClick();
    setGroups(prev => prev.map(g => {
      if (g.members.length === 0) return g;
      const randomLeader = g.members[Math.floor(Math.random() * g.members.length)];
      return { ...g, leaderId: randomLeader.id };
    }));
  };

  // Toggle specific student as leader
  const handleToggleLeader = (groupId: string, studentId: string) => {
    soundFx.playSoftClick();
    setGroups(prev => prev.map(g => {
      if (g.id !== groupId) return g;
      return {
        ...g,
        leaderId: g.leaderId === studentId ? undefined : studentId
      };
    }));
  };

  // Rename group
  const handleRenameGroup = (groupId: string, newName: string) => {
    setGroups(prev => prev.map(g => g.id === groupId ? { ...g, name: newName } : g));
  };

  // Move student between groups
  const handleMoveStudent = (toGroupId: string) => {
    if (!selectedStudentToMove || selectedStudentToMove.fromGroupId === toGroupId) {
      setSelectedStudentToMove(null);
      return;
    }
    soundFx.playSoftClick();
    setGroups(prev => {
      const student = selectedStudentToMove.student;
      return prev.map(g => {
        if (g.id === selectedStudentToMove.fromGroupId) {
          return {
            ...g,
            members: g.members.filter(m => m.id !== student.id),
            leaderId: g.leaderId === student.id ? undefined : g.leaderId
          };
        }
        if (g.id === toGroupId) {
          return {
            ...g,
            members: [...g.members, student]
          };
        }
        return g;
      });
    });
    setSelectedStudentToMove(null);
  };

  // Copy formatted text
  const handleCopyResults = () => {
    if (groups.length === 0) return;
    const text = groups.map(g => {
      const membersText = g.members.map(m => {
        const isLeader = g.leaderId === m.id;
        const seat = m.seatNo ? `(${m.seatNo}號)` : '';
        return `${m.name}${seat}${isLeader ? ' [組長]' : ''}`;
      }).join('、');
      return `【${g.name}】(${g.members.length}人)\n${membersText}`;
    }).join('\n\n');

    navigator.clipboard.writeText(text);
    setCopied(true);
    soundFx.playSoftClick();
    setTimeout(() => setCopied(false), 2000);
  };

  // Export CSV
  const handleExportCsv = () => {
    if (groups.length === 0) return;
    let csv = '\uFEFF組別,座號,姓名,是否為組長\n';
    groups.forEach(g => {
      g.members.forEach(m => {
        const isLeader = g.leaderId === m.id ? '是' : '否';
        csv += `"${g.name}",${m.seatNo || ''},"${m.name}","${isLeader}"\n`;
      });
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `分組名單_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    soundFx.playSoftClick();
  };

  // Print view
  const handlePrint = () => {
    window.print();
  };

  if (students.length === 0) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <div className="w-16 h-16 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto mb-4">
          <AlertCircle className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-slate-800 mb-2">尚未匯入學生名單</h2>
        <p className="text-sm text-slate-500 mb-6">
          分組前需要學生名單，請先前往名單管理上傳 CSV 或貼上姓名。
        </p>
        <button
          id="btn-goto-roster-grouping"
          onClick={onNavigateToRoster}
          className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold shadow-xs transition"
        >
          前往名單管理
        </button>
      </div>
    );
  }

  // Pre-calculate estimated group count for preview
  const estimatedGroups = groupMode === 'bySize'
    ? remainderStrategy === 'distribute'
      ? Math.floor(students.length / Math.max(1, targetSize)) || 1
      : Math.ceil(students.length / Math.max(1, targetSize))
    : targetGroupCount;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Control Panel Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 mb-8 print:hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          {/* Settings Section */}
          <div className="flex-1 space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                分組模式：
              </span>
              <div className="inline-flex bg-slate-100 p-1 rounded-xl border border-slate-200">
                <button
                  id="btn-group-by-size"
                  onClick={() => {
                    soundFx.playSoftClick();
                    setGroupMode('bySize');
                  }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                    groupMode === 'bySize'
                      ? 'bg-white text-emerald-700 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  依每組人數分組
                </button>
                <button
                  id="btn-group-by-count"
                  onClick={() => {
                    soundFx.playSoftClick();
                    setGroupMode('byCount');
                  }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                    groupMode === 'byCount'
                      ? 'bg-white text-emerald-700 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  依總組數分組
                </button>
              </div>

              {/* Leader assignment toggle */}
              <label className="flex items-center gap-2 text-xs font-semibold text-slate-700 cursor-pointer ml-auto">
                <input
                  id="checkbox-assign-leaders"
                  type="checkbox"
                  checked={assignLeaders}
                  onChange={(e) => setAssignLeaders(e.target.checked)}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500 w-4 h-4"
                />
                <Crown className="w-3.5 h-3.5 text-amber-500" />
                自動隨機指定各組組長
              </label>
            </div>

            {/* Steppers and Details */}
            <div className="flex flex-wrap items-center gap-6 pt-2">
              {groupMode === 'bySize' ? (
                <div className="flex items-center gap-3">
                  <label htmlFor="input-target-size" className="text-sm font-bold text-slate-800">
                    每組人數：
                  </label>
                  <div className="flex items-center border border-slate-300 rounded-xl overflow-hidden bg-slate-50">
                    <button
                      id="btn-decrease-size"
                      onClick={() => setTargetSize(Math.max(2, targetSize - 1))}
                      className="px-3 py-1.5 hover:bg-slate-200 text-slate-700 font-bold transition"
                    >
                      -
                    </button>
                    <input
                      id="input-target-size"
                      type="number"
                      min={2}
                      max={students.length}
                      value={targetSize}
                      onChange={(e) => setTargetSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
                      className="w-14 text-center py-1.5 font-bold text-sm bg-white border-x border-slate-300 focus:outline-none"
                    />
                    <button
                      id="btn-increase-size"
                      onClick={() => setTargetSize(Math.min(students.length, targetSize + 1))}
                      className="px-3 py-1.5 hover:bg-slate-200 text-slate-700 font-bold transition"
                    >
                      +
                    </button>
                  </div>
                  <span className="text-xs text-slate-500">人 / 組</span>

                  {/* Remainder option */}
                  <div className="flex items-center gap-2 ml-4 text-xs text-slate-600">
                    <span>若有餘數：</span>
                    <select
                      id="select-remainder-strategy"
                      value={remainderStrategy}
                      onChange={(e) => setRemainderStrategy(e.target.value as RemainderStrategy)}
                      className="px-2 py-1 rounded-lg border border-slate-300 bg-white text-xs font-medium text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                    >
                      <option value="distribute">平均併入各組（每組人數相差最多1人）</option>
                      <option value="newGroup">餘數獨立成額外一組</option>
                    </select>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <label htmlFor="input-target-groups" className="text-sm font-bold text-slate-800">
                    預計分為：
                  </label>
                  <div className="flex items-center border border-slate-300 rounded-xl overflow-hidden bg-slate-50">
                    <button
                      id="btn-decrease-groups"
                      onClick={() => setTargetGroupCount(Math.max(2, targetGroupCount - 1))}
                      className="px-3 py-1.5 hover:bg-slate-200 text-slate-700 font-bold transition"
                    >
                      -
                    </button>
                    <input
                      id="input-target-groups"
                      type="number"
                      min={2}
                      max={students.length}
                      value={targetGroupCount}
                      onChange={(e) => setTargetGroupCount(Math.max(1, parseInt(e.target.value, 10) || 1))}
                      className="w-14 text-center py-1.5 font-bold text-sm bg-white border-x border-slate-300 focus:outline-none"
                    />
                    <button
                      id="btn-increase-groups"
                      onClick={() => setTargetGroupCount(Math.min(students.length, targetGroupCount + 1))}
                      className="px-3 py-1.5 hover:bg-slate-200 text-slate-700 font-bold transition"
                    >
                      +
                    </button>
                  </div>
                  <span className="text-xs text-slate-500">組</span>
                </div>
              )}

              {/* Live Preview info */}
              <div className="text-xs text-slate-500 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
                全班 <span className="font-bold text-slate-800">{students.length}</span> 人，預計分約{' '}
                <span className="font-bold text-emerald-700">{estimatedGroups}</span> 組
              </div>
            </div>
          </div>

          {/* Action Trigger Button */}
          <div className="shrink-0 flex items-center">
            <button
              id="btn-generate-groups"
              onClick={generateGroups}
              className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-base shadow-md shadow-emerald-200 hover:shadow-emerald-300 transition flex items-center justify-center gap-2"
            >
              <Shuffle className="w-5 h-5" />
              <span>{groups.length > 0 ? '重新隨機分組' : '立即隨機分組'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Grouping Results Display */}
      {groups.length === 0 ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-xs">
          <div className="w-16 h-16 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-4">
            <Users className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-slate-800 mb-1">準備就緒，點擊上方按鈕開始分組</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto mb-6">
            系統將使用公平的隨機演算法，依據所設定的人數將全班 {students.length} 位學生平均分派至各組。
          </p>
          <button
            onClick={generateGroups}
            className="px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm shadow-xs transition"
          >
            立即產生分組
          </button>
        </div>
      ) : (
        <div>
          {/* Sub-toolbar: Actions on results */}
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6 print:hidden">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-slate-900">
                分組結果展示
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800">
                共 {groups.length} 組
              </span>
              {selectedStudentToMove && (
                <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-md border border-amber-300">
                  正在移動：{selectedStudentToMove.student.name}（請點選目標組別以移入）
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {assignLeaders && (
                <button
                  id="btn-reassign-leaders"
                  onClick={handleReassignLeaders}
                  title="重新隨機指定各組組長"
                  className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-100 transition flex items-center gap-1.5"
                >
                  <Crown className="w-3.5 h-3.5 text-amber-500" />
                  重抽組長
                </button>
              )}

              <button
                id="btn-copy-groups"
                onClick={handleCopyResults}
                className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-100 transition flex items-center gap-1.5"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? '已複製！' : '複製文字結果'}
              </button>

              <button
                id="btn-export-groups-csv"
                onClick={handleExportCsv}
                className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-100 transition flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5" />
                匯出 CSV
              </button>

              <button
                id="btn-print-groups"
                onClick={handlePrint}
                className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-100 transition flex items-center gap-1.5"
              >
                <Printer className="w-3.5 h-3.5" />
                列印 / 投影
              </button>
            </div>
          </div>

          {/* Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
            {groups.map((group) => {
              const palette = GROUP_PALETTES[group.colorIndex % GROUP_PALETTES.length];
              const isTargetForMove = selectedStudentToMove && selectedStudentToMove.fromGroupId !== group.id;

              return (
                <div
                  key={group.id}
                  className={`bg-white rounded-2xl border ${palette.border} shadow-xs overflow-hidden flex flex-col transition-all hover:shadow-md ${
                    isTargetForMove ? 'ring-2 ring-emerald-500 ring-offset-2' : ''
                  }`}
                >
                  {/* Card Header */}
                  <div className={`${palette.header} px-4 py-3 flex items-center justify-between`}>
                    <input
                      type="text"
                      value={group.name}
                      onChange={(e) => handleRenameGroup(group.id, e.target.value)}
                      className="bg-transparent text-white font-bold text-sm tracking-wide focus:outline-none focus:bg-white/20 px-1.5 py-0.5 rounded transition w-2/3"
                      title="點擊可修改組名"
                    />
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs bg-white/20 backdrop-blur-xs px-2 py-0.5 rounded-full font-bold">
                        {group.members.length} 人
                      </span>
                      {isTargetForMove && (
                        <button
                          onClick={() => handleMoveStudent(group.id)}
                          className="text-[11px] bg-white text-emerald-800 font-bold px-2 py-0.5 rounded hover:bg-slate-100 shadow-2xs"
                        >
                          移至此組
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Members List */}
                  <div className="p-4 flex-1 divide-y divide-slate-100">
                    {group.members.length === 0 ? (
                      <div className="py-6 text-center text-xs text-slate-400 italic">
                        此組尚無成員
                      </div>
                    ) : (
                      group.members.map((member) => {
                        const isLeader = group.leaderId === member.id;
                        const isBeingMoved = selectedStudentToMove?.student.id === member.id;

                        return (
                          <div
                            key={member.id}
                            className={`py-2 px-1 flex items-center justify-between group rounded-lg transition ${
                              isBeingMoved ? 'bg-amber-50' : 'hover:bg-slate-50'
                            }`}
                          >
                            <div className="flex items-center gap-2 overflow-hidden">
                              <span className="w-5 h-5 rounded bg-slate-100 text-slate-600 text-[11px] font-bold flex items-center justify-center shrink-0">
                                {member.seatNo || '-'}
                              </span>
                              <span className="text-sm font-semibold text-slate-800 truncate">
                                {member.name}
                              </span>
                              {isLeader && (
                                <span 
                                  title="此組組長"
                                  className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300 shrink-0"
                                >
                                  <Crown className="w-3 h-3 text-amber-600" />
                                  組長
                                </span>
                              )}
                            </div>

                            {/* Actions on student: Toggle Leader, Move Group */}
                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition print:hidden">
                              <button
                                onClick={() => handleToggleLeader(group.id, member.id)}
                                title={isLeader ? '取消組長' : '設為組長'}
                                className={`p-1 rounded text-xs transition ${
                                  isLeader ? 'text-amber-600 bg-amber-50' : 'text-slate-400 hover:text-amber-500'
                                }`}
                              >
                                <Crown className="w-3.5 h-3.5" />
                              </button>

                              <button
                                onClick={() => {
                                  if (isBeingMoved) {
                                    setSelectedStudentToMove(null);
                                  } else {
                                    setSelectedStudentToMove({ student: member, fromGroupId: group.id });
                                  }
                                }}
                                title="更換組別"
                                className={`p-1 rounded text-xs transition ${
                                  isBeingMoved ? 'text-amber-700 bg-amber-100' : 'text-slate-400 hover:text-slate-700'
                                }`}
                              >
                                <ArrowRightLeft className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
