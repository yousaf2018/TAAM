from PyQt6.QtCore import QThread, pyqtSignal
from backend.analysis_engine import AnalysisEngine
import pandas as pd
import os

class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, experimental_groups, arenas, arena_configs, config, workspace):
        super().__init__()
        self.groups = experimental_groups
        self.arenas = arenas
        self.arena_configs = arena_configs
        self.config = config
        self.workspace = workspace

    def run(self):
        all_sums, all_kins, stems = [], [], []
        files = [p for paths in self.groups.values() for p in paths]
        
        try:
            for i, p in enumerate(files):
                g_name = next(g for g, paths in self.groups.items() if p in paths)
                stems.append(os.path.splitext(os.path.basename(p))[0])
                
                # Engine Call
                sums, kins = AnalysisEngine.calculate_behavior(p, self.config, self.arenas, self.arena_configs, self.log_signal)
                
                for s in sums:
                    s['Group'] = g_name
                    s['Source'] = os.path.basename(p)
                    all_sums.append(s)
                
                kins['Group'] = g_name
                kins['Source'] = os.path.basename(p)
                all_kins.append(kins)
                
                self.progress_signal.emit(int(((i+1)/len(files))*100))

            if all_sums:
                output_name = "_".join(stems[:2]) + f"_n{len(stems)}_analysis.xlsx"
                save_path = os.path.join(self.workspace, output_name)
                self.log_signal.emit(f"[IO] Finalizing multi-sheet Excel file...")
                
                with pd.ExcelWriter(save_path) as writer:
                    pd.DataFrame(all_sums).to_excel(writer, sheet_name="Scientific_Summary", index=False)
                    pd.concat(all_kins).to_excel(writer, sheet_name="Detailed_Kinetics", index=False)
                
                self.finished_signal.emit(f"✅ Exported: {output_name}")
        except Exception as e:
            self.log_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")