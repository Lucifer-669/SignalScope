# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm as scipy_norm # For CDF

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QFrame,
    QStatusBar, QMessageBox, QSlider, QSpinBox, QDoubleSpinBox,
    QToolButton, QTextBrowser, QScrollArea, QSplitter, QComboBox,
    QSizePolicy, QFileDialog
)
from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QTextCursor, QMouseEvent
from PySide6.QtCore import Qt, Slot, QSize, Signal

# --- 信号传播模型等 ---
try:
    font_path = "C:/Windows/Fonts/simhei.ttf" # 字体文件路径，用于Matplotlib显示中文
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except FileNotFoundError:
    print("警告：指定的字体文件未找到，Matplotlib 中文可能无法正常显示。")
plt.rcParams['axes.unicode_minus'] = False # 解决Matplotlib坐标轴负号显示问题


def signal_strength_model(distance, transmit_power_dbm=30, frequency_mhz=2600,
                          antenna_gain_dbi=15, environment_type='urban_dense'):
    if distance <= 0:
        return transmit_power_dbm - 200

    if frequency_mhz <= 0:
        return transmit_power_dbm - 300

    distance_km = distance / 1000.0
    if distance_km == 0: distance_km = 0.000001

    fspl_m = 20 * np.log10(distance) + 20 * np.log10(frequency_mhz) - 27.55

    additional_loss = 0
    clutter_factor = 0
    path_loss_exponent_factor = 0

    if environment_type == 'urban_dense':
        clutter_factor = 20 + 0.1 * (frequency_mhz / 1000) * np.log10(distance + 1)
        path_loss_exponent_factor = 18 * np.log10(distance_km + 0.001)
        additional_loss = clutter_factor + path_loss_exponent_factor
    elif environment_type == 'urban_macro':
        clutter_factor = 15 + 0.05 * (frequency_mhz / 1000) * np.log10(distance + 1)
        path_loss_exponent_factor = 12 * np.log10(distance_km + 0.001)
        additional_loss = clutter_factor + path_loss_exponent_factor
    elif environment_type == 'suburban':
        clutter_factor = 10
        path_loss_exponent_factor = 8 * np.log10(distance_km + 0.001)
        additional_loss = clutter_factor + path_loss_exponent_factor
    elif environment_type == 'rural_macro':
        clutter_factor = 3
        path_loss_exponent_factor = 5 * np.log10(distance_km + 0.001)
        additional_loss = clutter_factor + path_loss_exponent_factor

    additional_loss = max(0, additional_loss)
    path_loss = fspl_m + additional_loss
    eirp_dbm = transmit_power_dbm + antenna_gain_dbi
    received_power_dbm = eirp_dbm - path_loss
    return received_power_dbm


def create_grid(x_min, x_max, y_min, y_max, step):
    if step <= 0:
        raise ValueError("网格步长必须为正数。")
    x = np.arange(x_min, x_max + step, step)
    y = np.arange(y_min, y_max + step, step)
    if x_min >= x_max: x = np.array([x_min])
    if y_min >= y_max: y = np.array([y_min])
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def simulate_signal_strength(grid, base_station_params, transmit_power_dbm,
                             frequency_mhz, antenna_gain_dbi, environment_type):
    xx, yy = grid
    bs_x, bs_y = base_station_params['x'], base_station_params['y']

    if xx.shape == (1, 1) and yy.shape == (1, 1):
        distances = np.array([[np.sqrt((xx[0, 0] - bs_x) ** 2 + (yy[0, 0] - bs_y) ** 2)]])
    else:
        distances = np.sqrt((xx - bs_x) ** 2 + (yy - bs_y) ** 2)

    v_signal_strength = np.vectorize(signal_strength_model,
                                     excluded=['transmit_power_dbm', 'frequency_mhz', 'antenna_gain_dbi',
                                               'environment_type'])
    strengths = v_signal_strength(distances,
                                  transmit_power_dbm=transmit_power_dbm,
                                  frequency_mhz=frequency_mhz,
                                  antenna_gain_dbi=antenna_gain_dbi,
                                  environment_type=environment_type)
    return strengths

# --- Helper for Sliders ---
def create_slider_spinbox_combo(param_label_text, info_key, emoji, min_val, max_val, default_val, step_val, decimals=1, parent_app=None):
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(8)

    if parent_app: # For clickable label to connect to app's display update
        clickable_label = ClickableLabel(param_label_text, info_key)
        clickable_label.clicked.connect(parent_app.update_param_info_display)
    else:
        clickable_label = QLabel(param_label_text) # Fallback

    label_widget = QWidget()
    label_layout = QHBoxLayout(label_widget)
    label_layout.setContentsMargins(0,0,0,0)
    if emoji:
        icon_label = QLabel(emoji)
        icon_label.setObjectName("EmojiIconLabel")
        label_layout.addWidget(icon_label)
    label_layout.addWidget(clickable_label)
    label_layout.addStretch()

    slider_multiplier = 10**decimals
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(int(min_val * slider_multiplier), int(max_val * slider_multiplier))
    slider.setValue(int(default_val * slider_multiplier))
    slider.setSingleStep(int(step_val * slider_multiplier))
    slider.setTickInterval(int((max_val-min_val)/5 * slider_multiplier)) # Example tick interval
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)

    spinbox = QDoubleSpinBox()
    spinbox.setDecimals(decimals)
    spinbox.setRange(min_val, max_val)
    spinbox.setValue(default_val)
    spinbox.setSingleStep(step_val)
    spinbox.setObjectName("InputField")
    spinbox.setMinimumWidth(80) # Ensure spinbox is wide enough

    # Connect slider and spinbox
    slider.valueChanged.connect(lambda val: spinbox.setValue(val / slider_multiplier))
    spinbox.valueChanged.connect(lambda val: slider.setValue(int(val * slider_multiplier)))

    layout.addWidget(label_widget, 1) # Label part takes less stretch
    layout.addWidget(slider, 3)     # Slider takes more stretch
    layout.addWidget(spinbox, 1)   # Spinbox takes less stretch

    return container, slider, spinbox


# --- 模型结束 ---

class ClickableLabel(QLabel):
    clicked = Signal(str)

    def __init__(self, text, key_for_info, parent=None):
        super().__init__(text, parent)
        self.key_for_info = key_for_info
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("ClickableParamLabel")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key_for_info)
        super().mousePressEvent(event)


class ProfessionalSignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高级5G覆盖仿真与分析平台 v1.2") # Version bump!
        self.setGeometry(50, 50, 1600, 980)

        self._main_widget = QWidget()
        self.setCentralWidget(self._main_widget)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self._main_widget)
        self.main_splitter.setObjectName("MainSplitter")

        self.setup_parameter_info_data() # Call this first to have explanations ready
        self._create_left_panel()
        self._create_right_panel()

        self.main_splitter.addWidget(self.left_panel_widget)
        self.main_splitter.addWidget(self.right_panel_widget)
        self.main_splitter.setSizes([520, 1080])

        main_layout_for_splitter = QHBoxLayout(self._main_widget)
        main_layout_for_splitter.addWidget(self.main_splitter)

        self._create_status_bar()
        self.apply_modern_stylesheet()
        self.current_simulation_results = None
        self.update_param_info_display("welcome")
        self.update_colormap_info_display(0) # Show info for default colormap

    def _create_left_panel(self):
        # ... (Code from previous _create_left_panel method, but with cmap_combo connection) ...
        self.left_panel_widget = QWidget()
        self.left_panel_widget.setObjectName("LeftPanel")
        left_v_layout = QVBoxLayout(self.left_panel_widget)
        left_v_layout.setContentsMargins(0, 0, 0, 0)

        param_scroll_area = QScrollArea()
        param_scroll_area.setWidgetResizable(True)
        param_scroll_area.setObjectName("ParameterScrollArea")

        param_input_widget = QWidget()
        param_input_widget.setObjectName("ParamInputCardWidget")
        param_input_layout = QVBoxLayout(param_input_widget)
        param_input_layout.setSpacing(15)
        param_input_layout.setContentsMargins(20, 20, 20, 20)

        self.default_param_values = {
            "env_type": 0, "bs_x": 0.0, "bs_y": 0.0, "tx_power": 40.0, "antenna_gain": 17.0,
            "frequency": 2600.0, "x_min": -500.0, "x_max": 500.0, "y_min": -500.0,
            "y_max": 500.0, "step": 10.0, "noise_floor": -110.0, "colormap": 0,
        }

        self.param_definitions = {
            "env_type": ("环境场景类型", self.default_param_values["env_type"], ["urban_dense", "urban_macro", "suburban", "rural_macro"], "env_type_desc", "🏞️"),
            "bs_x": ("基站 X 坐标 (米)", self.default_param_values["bs_x"], 10.0, "bs_x_desc", "📍"),
            "bs_y": ("基站 Y 坐标 (米)", self.default_param_values["bs_y"], 10.0, "bs_y_desc", "📍"),
            "tx_power": ("发射功率 (dBm)", self.default_param_values["tx_power"], "tx_power_desc", "🔋", [10.0, 60.0, 1.0, 1]),
            "antenna_gain": ("天线增益 (dBi)", self.default_param_values["antenna_gain"], "antenna_gain_desc", "📡", [0.0, 30.0, 0.5, 1]),
            "frequency": ("工作频率 (MHz)", self.default_param_values["frequency"], "frequency_desc", "📶", [600.0, 6000.0, 100.0, 0]),
            "noise_floor": ("背景噪声 (dBm)", self.default_param_values["noise_floor"], 1.0, "noise_floor_desc", "🤫"),
            "x_min": ("X 范围 Min (米)", self.default_param_values["x_min"], 10.0, "area_desc", "↔️"),
            "x_max": ("X 范围 Max (米)", self.default_param_values["x_max"], 10.0, "area_desc", "↔️"),
            "y_min": ("Y 范围 Min (米)", self.default_param_values["y_min"], 10.0, "area_desc", "↕️"),
            "y_max": ("Y 范围 Max (米)", self.default_param_values["y_max"], 10.0, "area_desc", "↕️"),
            "step": ("网格步长 (米)", self.default_param_values["step"], 1.0, "step_desc", "📏"),
        }
        self.param_inputs_widgets = {}
        groups_order = [
            ("场景与基站核心参数", ["env_type", "bs_x", "bs_y", "tx_power", "antenna_gain", "frequency", "noise_floor"]),
            ("模拟区域与精度", ["x_min", "x_max", "y_min", "y_max", "step"])
        ]
        self.env_type_map = {"urban_dense": "密集城区", "urban_macro": "一般城区", "suburban": "郊区", "rural_macro": "农村/开阔地"}

        for group_title, param_keys in groups_order:
            group_box = QGroupBox(group_title)
            group_box.setObjectName("ParameterGroup")
            group_layout = QGridLayout(group_box)
            group_layout.setHorizontalSpacing(10)
            group_layout.setVerticalSpacing(12)
            row = 0
            for key in param_keys:
                if key not in self.param_definitions: continue
                param_data = self.param_definitions[key]
                if len(param_data) == 5 and isinstance(param_data[-1], list):
                    label_text, default_val, info_key, emoji, slider_params = param_data
                    min_s, max_s, step_s, dec_s = slider_params
                    combo_widget, slider, spinbox = create_slider_spinbox_combo(
                        label_text, info_key, emoji, min_s, max_s, default_val, step_s, dec_s, parent_app=self)
                    group_layout.addWidget(combo_widget, row, 0, 1, 2)
                    self.param_inputs_widgets[key] = spinbox
                    self.param_inputs_widgets[key + "_slider"] = slider
                else:
                    if key == "env_type": label_text, default_idx, options_keys, info_key, emoji_icon = param_data
                    else: label_text, default_val, step_v, info_key, emoji_icon = param_data
                    clickable_label = ClickableLabel(label_text, info_key)
                    clickable_label.clicked.connect(self.update_param_info_display)
                    label_container_widget = QWidget()
                    label_layout_h = QHBoxLayout(label_container_widget)
                    label_layout_h.setContentsMargins(0,0,0,0); label_layout_h.setSpacing(5)
                    if emoji_icon:
                        icon_label_s = QLabel(emoji_icon); icon_label_s.setObjectName("EmojiIconLabel")
                        label_layout_h.addWidget(icon_label_s)
                    label_layout_h.addWidget(clickable_label); label_layout_h.addStretch()
                    group_layout.addWidget(label_container_widget, row, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    if key == "env_type":
                        input_field = QComboBox(); input_field.setObjectName("ComboBoxField")
                        for option_key_item in options_keys: input_field.addItem(self.env_type_map.get(option_key_item, option_key_item), option_key_item)
                        input_field.setCurrentIndex(default_idx)
                    else:
                        is_float = isinstance(default_val, float) or isinstance(step_v, float)
                        if is_float: input_field = QDoubleSpinBox(); input_field.setDecimals(2); input_field.setRange(-1.0e18, 1.0e18)
                        else: input_field = QSpinBox(); input_field.setRange(-2147483647, 2147483647)
                        input_field.setValue(default_val); input_field.setSingleStep(step_v)
                        input_field.setObjectName("InputField")
                    self.param_inputs_widgets[key] = input_field
                    group_layout.addWidget(input_field, row, 1)
                row += 1
            param_input_layout.addWidget(group_box)

        plot_options_group = QGroupBox("绘图选项")
        plot_options_group.setObjectName("ParameterGroup")
        plot_options_layout = QGridLayout(plot_options_group)
        cmap_label = QLabel("颜色映射:")
        self.cmap_combo = QComboBox()
        self.cmaps = {'Viridis (默认)': 'viridis', 'Plasma': 'plasma', 'Inferno': 'inferno', 'Magma': 'magma', 'RdYlBu (蓝黄红)': 'RdYlBu_r', 'RdYlGn (绿黄红)': 'RdYlGn_r'}
        for name in self.cmaps.keys(): self.cmap_combo.addItem(name)
        self.cmap_combo.setCurrentIndex(self.default_param_values["colormap"])
        self.param_inputs_widgets["colormap"] = self.cmap_combo
        # Connect signal for colormap info display
        self.cmap_combo.currentIndexChanged.connect(self.update_colormap_info_display) # <<<< NEW CONNECTION
        plot_options_layout.addWidget(cmap_label, 0, 0)
        plot_options_layout.addWidget(self.cmap_combo, 0, 1)
        param_input_layout.addWidget(plot_options_group)

        button_frame = QFrame(); button_frame.setObjectName("ButtonFrame")
        button_layout = QHBoxLayout(button_frame); button_layout.setSpacing(15)
        self.reset_button = QPushButton("🔄 重置参数"); self.reset_button.setObjectName("ResetButton")
        self.reset_button.clicked.connect(self.reset_parameters)
        self.simulate_button = QPushButton("🚀 运行仿真"); self.simulate_button.setObjectName("SimulateButtonPrimary")
        self.simulate_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.reset_button); button_layout.addWidget(self.simulate_button, 1)
        param_input_layout.addWidget(button_frame)
        param_input_layout.addStretch(1)

        param_scroll_area.setWidget(param_input_widget)
        left_v_layout.addWidget(param_scroll_area, 3)
        self.param_info_display = QTextBrowser(); self.param_info_display.setObjectName("ParamInfoDisplayCard")
        self.param_info_display.setOpenExternalLinks(True)
        left_v_layout.addWidget(self.param_info_display, 2)


    def _create_right_panel(self):
        # ... (Same as your last provided version) ...
        self.right_panel_widget = QWidget()
        self.right_panel_widget.setObjectName("RightPanel")
        right_v_layout = QVBoxLayout(self.right_panel_widget)
        right_v_layout.setContentsMargins(15, 15, 15, 15)
        right_v_layout.setSpacing(15)
        plot_card = QFrame(); plot_card.setObjectName("PlotCard")
        plot_layout = QVBoxLayout(plot_card); plot_layout.setContentsMargins(1,1,1,1)
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        toolbar_layout = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("MatplotlibToolbarModern")
        self.export_plot_button = QPushButton("导出覆盖图")
        # self.export_plot_button.setIcon(QIcon.fromTheme("document-save", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-save-32.png")))
        self.export_plot_button.clicked.connect(self.export_main_plot)
        toolbar_layout.addWidget(self.toolbar); toolbar_layout.addStretch(); toolbar_layout.addWidget(self.export_plot_button)
        plot_layout.addLayout(toolbar_layout)
        plot_layout.addWidget(self.canvas, 1)
        right_v_layout.addWidget(plot_card, 5)
        analysis_area_widget = QFrame(); analysis_area_widget.setObjectName("AnalysisSuperCard")
        analysis_v_layout = QVBoxLayout(analysis_area_widget)
        analysis_v_layout.setContentsMargins(0,0,0,0)
        self.analysis_display = QTextBrowser(); self.analysis_display.setObjectName("AnalysisDisplayCard")
        self.analysis_display.setReadOnly(True); self.analysis_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.analysis_display_initial_html = self.html_global_css + \
                                             "<div class='card-header'><h3>覆盖分析概要</h3></div>" + \
                                             "<div class='card-content'><p>仿真完成后，此处将展示详细的覆盖数据解读与专业分析。</p></div>"
        self.analysis_display.setHtml(self.analysis_display_initial_html)
        analysis_v_layout.addWidget(self.analysis_display, 2)
        cdf_plot_card = QFrame(); cdf_plot_card.setObjectName("PlotCard")
        cdf_plot_layout = QVBoxLayout(cdf_plot_card); cdf_plot_layout.setContentsMargins(5,5,5,5)
        self.cdf_figure = plt.figure(figsize=(6, 3.5), facecolor='white', dpi=90)
        self.cdf_canvas = FigureCanvas(self.cdf_figure)
        cdf_plot_layout.addWidget(self.cdf_canvas)
        analysis_v_layout.addWidget(cdf_plot_card, 3)
        right_v_layout.addWidget(analysis_area_widget, 4)


    def _create_status_bar(self):
        # ... (Same as before) ...
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("AppStatusBar")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("平台就绪：请在左侧配置参数，点击参数名称查看说明，然后运行仿真。")

    def setup_parameter_info_data(self):
        self.html_global_css = """
        <style>
            body { font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: 10pt; color: #34495E; line-height: 1.6; }
            .card-header { background-color: #F0F3F4; padding: 10px 15px; border-bottom: 1px solid #E0E5EA; border-top-left-radius: 7px; border-top-right-radius: 7px; margin:-1px -1px 0 -1px; }
            .card-header h3 { color: #2C3A47; font-size: 12pt; font-weight: bold; margin:0; padding:0; }
            .card-content { padding: 15px; }
            .card-content p { margin-bottom: 10px; }
            .card-content ul { margin-left: 20px; padding-left: 5px; }
            .card-content li { margin-bottom: 5px; }
            .param-highlight { color: #007BFF; font-weight: bold; }
            .pro-tip { background-color: #E9F7FD; border-left: 4px solid #007BFF; padding: 10px 15px; margin-top:12px; margin-bottom:8px; font-size:9.5pt; color: #2C3A47; border-radius: 4px;}
            table { width:100%; border-collapse: collapse; font-size: 9pt; margin-top:10px; }
            th { background-color:#5D6D7E; color:white; text-align:left; padding:6px 8px; font-weight:bold; }
            td { padding:6px 8px; border-bottom: 1px solid #E0E5EA; }
            tr:last-child td { border-bottom: none; }
            .rsrp-excellent { font-weight:bold; color:#1E8449; }
            .rsrp-good { font-weight:bold; color:#20C997; }
            .rsrp-fair { font-weight:bold; color:#FFC107; }
            .rsrp-marginal { font-weight:bold; color:#FD7E14; }
            .rsrp-poor { font-weight:bold; color:#DC3545; }
            .cmap-example-img { max-width: 250px; height: auto; display: block; margin-top: 10px; border: 1px solid #ddd; border-radius: 4px;}
        </style>
        """ # Added .cmap-example-img style

        self.parameter_explanations = {
            "welcome": self.html_global_css + "<div class='card-header'><h3>参数说明</h3></div><div class='card-content'><p>欢迎使用5G覆盖仿真平台！请点击左侧面板中的<span class='param-highlight'>参数名称</span> (例如“发射功率”)，此处将显示该参数的详细定义、在5G网络中的作用，以及调整它对网络覆盖的典型影响。</p><p><b>新增功能:</b><ul><li>主要参数（功率、增益、频率）增加了滑块调节。</li><li>可以选择热力图的颜色映射方案 (点击下拉框查看说明)。</li><li>分析报告下方增加了RSRP累积分布函数(CDF)图。</li><li>增加了背景噪声参数输入。</li><li>可导出覆盖图。</li></ul></p></div>",
            "env_type_desc": self.html_global_css + """<div class='card-header'><h3>🏞️ 环境场景类型</h3></div>
                               <div class='card-content'>
                               <p><b>定义：</b>选择基站所处的典型无线传播环境。不同的环境（如密集城区、一般城区、郊区、农村）对无线信号的传播和衰减有显著不同的影响。</p>
                               <p><b>作用与影响：</b>
                                   <ul>
                                       <li><b>密集城区 (Urban Dense):</b> 建筑物高大密集，街道狭窄。信号衰减快，多径效应严重，穿透损耗大。覆盖范围通常较小，需要更密集的基站部署。</li>
                                       <li><b>一般城区 (Urban Macro):</b> 建筑物较密集，但可能有较开阔的街道或区域。衰减和多径效应仍然显著。</li>
                                       <li><b>郊区 (Suburban):</b> 建筑物相对稀疏，高度较低，有较多开阔空间。信号传播条件优于城区，覆盖范围较大。</li>
                                       <li><b>农村/开阔地 (Rural Macro):</b> 地势平坦，障碍物少。信号传播接近自由空间，路径损耗最小，单站覆盖范围最大。</li>
                                   </ul>
                               </p>
                               <p class='pro-tip'><b>专业提示：</b>本仿真器使用简化的模型调整来示意不同环境的影响。实际网络规划会采用更精确的传播模型（如Okumura-Hata, COST231-Hata, 3GPP TR 38.901等），这些模型会考虑基站和终端天线高度、建筑物平均高度、街道宽度等更多细节参数。</p>
                               </div>""",
            "bs_x_desc": self.html_global_css + """<div class='card-header'><h3>📍 基站 X/Y 坐标 (米)</h3></div>
                           <div class='card-content'>
                           <p><b>定义：</b>基站在模拟地图上的水平 (X) 和垂直 (Y) 位置。这是所有距离和信号强度计算的<b>几何中心</b>。</p>
                           <p><b>作用与影响：</b>此参数直接决定了信号覆盖区域的<b>地理中心位置</b>。改变基站坐标会整体平移信号覆盖图。在实际网络规划中，基站选址是首要考虑因素，需结合目标覆盖区域的地形地貌、建筑物分布、用户密度以及与周边基站的协同关系等。</p>
                           <p class='pro-tip'><b>专业提示：</b>不合理的基站选址可能导致覆盖空洞、越区覆盖（干扰邻小区）或资源浪费。</p>
                           </div>""",
            "bs_y_desc": self.html_global_css + """<div class='card-header'><h3>📍 基站 X/Y 坐标 (米)</h3></div>
                           <div class='card-content'>
                           <p>（同基站X坐标）</p>
                           <p><b>定义：</b>基站在模拟地图上的垂直 (Y) 位置。</p>
                           </div>""",
            "tx_power_desc": self.html_global_css + """<div class='card-header'><h3>🔋 发射功率 (dBm)</h3></div>
                                <div class='card-content'>
                                <p><b>定义：</b>基站发射天线端口输出的无线电信号能量强度，通常指单个载波或一个小区总的<b>传导功率</b>。单位 dBm (分贝毫瓦) 是一个对数单位，0 dBm = 1 mW。</p>
                                <p><b>作用与影响：</b>
                                    <ul>
                                        <li><b>增加发射功率：</b>会显著<b>扩大信号覆盖半径</b>，并提升小区边缘区域的信号强度 (RSRP) 和信噪比 (SINR)，从而改善边缘用户的下载速率和连接稳定性。</li>
                                        <li><b>减少发射功率：</b>则会<b>缩小覆盖半径</b>。</li>
                                    </ul>
                                </p>
                                <p class='pro-tip'><b>专业提示：</b>发射功率并非越大越好。虽然提高功率能扩大覆盖，但过高的功率会增加对邻近小区的干扰，可能导致整个网络性能下降（“呼吸效应”的体现之一）。5G宏基站典型单通道传导功率范围约 <b>20W (43dBm) 至 80W (49dBm)</b>。总功率会根据配置的通道数和带宽而变化。功率控制是网络优化的重要手段。</p>
                                </div>""",
            "antenna_gain_desc": self.html_global_css + """<div class='card-header'><h3>📡 天线增益 (dBi)</h3></div>
                                   <div class='card-content'>
                                   <p><b>定义：</b>衡量天线将输入功率有效地<b>汇聚并辐射到特定方向</b>的能力，相对于一个理想的、无损耗的全向点源天线 (Isotropic Antenna) 的增益。单位 dBi。</p>
                                   <p><b>作用与影响：</b>天线通过其方向性图样将能量集中在主辐射瓣内。
                                       <li><b>增加天线增益：</b>在天线主瓣方向上能显著提升<b>等效全向辐射功率 (EIRP)</b>，从而扩大覆盖或改善特定方向的信号质量，而无需增加基站的实际发射功率。</li>
                                       <li>天线增益通常与其<b>波束宽度</b>成反比：增益越高，主瓣波束越窄，能量越集中。</li>
                                   </p>
                                   <p class='pro-tip'><b>专业提示：</b>EIRP (dBm) = 发射功率(dBm) + 天线增益(dBi) – 线缆损耗(dB)。EIRP是链路预算中真正决定远端信号强度的关键。5G Massive MIMO天线系统具有非常高的等效增益和灵活的波束赋形能力，是实现精准覆盖和容量提升的核心技术之一。典型5G宏站天线增益在 <b>15 dBi 到 25 dBi</b> 之间。</p>
                                   </div>""",
            "frequency_desc": self.html_global_css + """<div class='card-header'><h3>📶 工作频率 (MHz)</h3></div>
                               <div class='card-content'>
                               <p><b>定义：</b>基站发射无线电波所使用的中心频率。5G网络在全球范围内分配了多个频段，通常分为FR1 (Sub-6GHz，如700MHz, 2.1GHz, 2.6GHz, 3.5GHz, 4.9GHz) 和FR2 (毫米波，如26GHz, 28GHz, 39GHz)。</p>
                               <p><b>作用与影响：</b>频率是影响无线电波<b>传播特性</b>的核心物理因素。
                                   <ul>
                                       <li><b>较低频率 (如700MHz，被誉为“黄金频段”)：</b>具有更强的<b>穿透能力</b>（穿墙）和<b>绕射能力</b>（绕过障碍物），以及更小的<b>路径损耗</b>。因此，低频段能实现更广的覆盖范围，尤其适合农村地区广域覆盖和城市深度室内覆盖。</li>
                                       <li><b>较高频率 (如3.5GHz，毫米波)：</b>拥有更丰富的<b>带宽资源</b>，能够支持更高的数据传输速率和网络容量。但其穿透和绕射能力较弱，路径损耗大，导致覆盖范围相对较小，更适用于城市热点区域、场馆等场景的容量补充和高速率服务。</li>
                                   </ul>
                               </p>
                               <p class='pro-tip'><b>专业提示：</b>自由空间路径损耗 (FSPL) 与频率的平方成正比 (FSPL ∝ f²)。这意味着，在相同距离下，频率越高，信号衰减越严重。因此，不同频段的组网策略和站址密度会有显著差异。</p>
                               </div>""",
            "noise_floor_desc": self.html_global_css + """<div class='card-header'><h3>🤫 背景噪声 (dBm)</h3></div>
                                   <div class='card-content'>
                                   <p><b>定义：</b>接收机在没有期望信号存在时，自身产生的内部噪声以及从外部环境接收到的所有不需要的射频能量的总和，通常表示为一个等效的噪声功率电平。</p>
                                   <p><b>作用与影响：</b>背景噪声是限制通信系统性能的关键因素。它直接影响信噪比 (SNR) 和信干噪比 (SINR)。
                                       <li><b>较低的噪声基底：</b>意味着接收机可以检测到更微弱的信号，从而可能扩大有效覆盖范围或在给定信号强度下获得更好的数据速率。</li>
                                       <li><b>较高的噪声基底：</b>会淹没微弱信号，降低SNR/SINR，导致通信质量下降，有效覆盖范围缩小。</li>
                                   </p>
                                   <p class='pro-tip'><b>专业提示：</b>典型的5G NR接收机在室温下的热噪声基底约为 -174 dBm/Hz。对于一个20MHz带宽的信道，总的热噪声约为 -174 dBm/Hz + 10*log10(20*10^6 Hz) ≈ -101 dBm。再加上接收机自身的噪声系数 (Noise Figure, NF)，例如5-7 dB，则实际的噪声基底可能在 <b>-96 dBm 至 -94 dBm</b> 左右。此参数对于后续计算SINR至关重要。</p>
                                   </div>""",
            "area_desc": self.html_global_css + """<div class='card-header'><h3>↔️↕️ 模拟区域范围 (米)</h3></div>
                            <div class='card-content'>
                            <p><b>定义：</b>设定进行信号覆盖仿真的二维地理区域的边界 (X轴和Y轴的最小、最大坐标)。</p>
                            <p><b>作用与影响：</b>此参数本身不直接影响无线电波的物理传播特性，但它决定了仿真程序<b>计算和显示信号强度分布图的可视范围</b>。合理的范围设置有助于聚焦于目标分析区域。</p>
                            </div>""",
            "step_desc": self.html_global_css + """<div class='card-header'><h3>📏 网格步长 (米)</h3></div>
                           <div class='card-content'>
                           <p><b>定义：</b>在选定的模拟区域内，进行信号强度计算的<b>空间采样点间距</b>。步长越小，计算点越密集，形成的网格越精细。</p>
                           <p><b>作用与影响：</b>
                               <ul>
                                   <li><b>较小步长：</b>能够更精确地捕捉和描绘信号强度的细微变化，生成的覆盖图更平滑、细节更丰富。但计算量会显著增加 (与步长平方成反比)，导致仿真时间变长。</li>
                                   <li><b>较大步长：</b>计算速度快，能快速得到大致的覆盖轮廓。但可能丢失局部区域的信号细节，图像显得粗糙或出现“马赛克”效应。</li>
                               </ul>
                           </p>
                           <p class='pro-tip'><b>专业提示：</b>选择合适的步长需要在仿真精度和计算效率之间进行权衡。通常，在初步探索或大范围概览时可使用较大步长，在需要对特定小区域进行精细分析时再适当减小步长。</p>
                           </div>""",
        }
        self.colormap_explanations = {
            'Viridis (默认)': self.html_global_css + """<div class='card-header'><h3>Viridis 颜色映射</h3></div>
                                 <div class='card-content'>
                                 <p><b>特点：</b>感知统一、色盲友好、亮度单调。从深紫色/蓝色 (低RSRP值) 过渡到绿色，再到亮黄色 (高RSRP值)。</p>
                                 <p><b>适用场景：</b>科学数据可视化，能准确反映数据变化，避免视觉偏差。是Matplotlib的默认颜色图之一。</p>
                                 <!-- <img class='cmap-example-img' src='path/to/viridis_example.png' alt='Viridis示例'></img> -->
                                 </div>""",
            'Plasma': self.html_global_css + """<div class='card-header'><h3>Plasma 颜色映射</h3></div>
                         <div class='card-content'>
                         <p><b>特点：</b>感知统一、色盲友好。从深蓝色 (低值) 过渡到洋红色，再到亮黄色 (高值)。</p>
                         <p><b>适用场景：</b>与Viridis类似，提供了另一种鲜明的色觉体验。</p>
                         <!-- <img class='cmap-example-img' src='path/to/plasma_example.png' alt='Plasma示例'></img> -->
                         </div>""",
            'Inferno': self.html_global_css + """<div class='card-header'><h3>Inferno 颜色映射</h3></div>
                          <div class='card-content'>
                          <p><b>特点：</b>感知统一。从黑色 (低值) 过渡到红色、橙色，再到亮黄色 (高值)。</p>
                          <p><b>适用场景：</b>常用于表示强度或温度等物理量，颜色过渡较为激烈。</p>
                          <!-- <img class='cmap-example-img' src='path/to/inferno_example.png' alt='Inferno示例'></img> -->
                          </div>""",
            'Magma': self.html_global_css + """<div class='card-header'><h3>Magma 颜色映射</h3></div>
                         <div class='card-content'>
                         <p><b>特点：</b>感知统一。从黑色 (低值) 过渡到洋红色、橙色，再到近白色 (高值)。</p>
                         <p><b>适用场景：</b>与Inferno类似，但高值端更亮，对比度更强。</p>
                         <!-- <img class='cmap-example-img' src='path/to/magma_example.png' alt='Magma示例'></img> -->
                         </div>""",
            'RdYlBu (蓝黄红)': self.html_global_css + """<div class='card-header'><h3>RdYlBu (蓝-黄-红) 颜色映射</h3></div>
                         <div class='card-content'>
                         <p><b>特点：</b>发散型颜色映射。这里使用反向 (`_r`)，即蓝色(低RSRP值)-黄色(中RSRP值)-红色(高RSRP值)。</p>
                         <p><b>适用场景：</b>通常用于数据有自然中心点或需要突出两端极值的情况。对于RSRP这种单调变化的信号强度，如果希望红色代表强信号，蓝色代表弱信号，此映射适用。</p>
                         <!-- <img class='cmap-example-img' src='path/to/rdylbu_example.png' alt='RdYlBu示例'></img> -->
                         </div>""",
            'RdYlGn (绿黄红)': self.html_global_css + """<div class='card-header'><h3>RdYlGn (绿-黄-红) 颜色映射</h3></div>
                         <div class='card-content'>
                         <p><b>特点：</b>发散型颜色映射，常用于表示好坏。这里使用反向 (`_r`)，即绿色(低RSRP值)-黄色(中RSRP值)-红色(高RSRP值)。</p>
                         <p><b>适用场景：</b>如果希望红色代表强信号，绿色代表弱信号，此映射适用。若希望绿色代表好信号（高RSRP），则需要调整颜色条的映射或使用非反向版本并注意颜色条方向。</p>
                         <!-- <img class='cmap-example-img' src='path/to/rdylgn_example.png' alt='RdYlGn示例'></img> -->
                         </div>"""
        }


    @Slot(str)
    def update_param_info_display(self, key):
        # ... (Same as before) ...
        if key in self.parameter_explanations:
            self.param_info_display.setHtml(self.parameter_explanations[key])
            self.param_info_display.moveCursor(QTextCursor.MoveOperation.Start)
        else:
            self.param_info_display.setHtml(
                self.html_global_css + f"<div class='card-content'><p>没有关于键 {key} 的信息。</p></div>")

    @Slot(int)
    def update_colormap_info_display(self, index):
        selected_cmap_name = self.cmap_combo.currentText()
        if selected_cmap_name in self.colormap_explanations:
            self.param_info_display.setHtml(self.colormap_explanations[selected_cmap_name])
            self.param_info_display.moveCursor(QTextCursor.MoveOperation.Start)
        else:
            # Fallback or do nothing
            self.update_param_info_display("welcome")


    def apply_modern_stylesheet(self):
        # ... (Same as your last provided version) ...
        qss = """
            QMainWindow { background-color: #F4F6F8; }
            QSplitter#MainSplitter::handle { background-color: #D1D8E0; width: 1px; }
            QWidget#LeftPanel { background-color: #FFFFFF; border-right: 1px solid #E0E5EA; }
            QScrollArea#ParameterScrollArea { border: none; background-color: transparent; }
            QWidget#ParamInputCardWidget { background-color: #FFFFFF; }
            QGroupBox#ParameterGroup {
                font-family: "Segoe UI Semibold", "Microsoft YaHei UI Semibold", sans-serif;
                font-size: 12pt; color: #2C3A47; border: 1px solid #E7E9EC;
                border-radius: 8px; margin-top: 12px; padding: 10px 15px 15px 15px;
                background-color: #FCFDFE;
            }
            QGroupBox#ParameterGroup::title {
                subcontrol-origin: margin; subcontrol-position: top left; padding: 5px 12px;
                margin-left: 10px; background-color: #5D6D7E; color: #FFFFFF;
                border-radius: 6px; font-size: 10pt; font-weight: bold;
            }
            QLabel#EmojiIconLabel { font-size: 12pt; padding-right: 4px; }
            QLabel#ClickableParamLabel {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 10pt; color: #34495E; padding: 4px 0px; text-decoration: none;
            }
            QLabel#ClickableParamLabel:hover { color: #007BFF; text-decoration: underline; }
            QSpinBox#InputField, QDoubleSpinBox#InputField, QComboBox#ComboBoxField {
                font-family: "Segoe UI", "Consolas", monospace;
                font-size: 10pt; padding: 7px 9px; border: 1px solid #CCD1D9;
                border-radius: 5px; background-color: #FDFEFE; color: #1F2933; min-height: 24px;
            }
            QComboBox#ComboBoxField QAbstractItemView { 
                border: 1px solid #CCD1D9; background-color: white;
                selection-background-color: #007BFF; font-size: 10pt;
            }
            QSpinBox#InputField:focus, QDoubleSpinBox#InputField:focus, QComboBox#ComboBoxField:focus {
                border-color: #007BFF; background-color: #FFFFFF; 
            }
            QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 8px; border-radius: 4px; }
            QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #56CCF2); border: 1px solid #777; height: 8px; border-radius: 4px; }
            QSlider::add-page:horizontal { background: #fff; border: 1px solid #777; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.6 #e1e1e1, stop:0.776 #3D72A4); border: 1px solid #777; width: 16px; margin-top: -4px; margin-bottom: -4px; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #007BFF; border-color: #0056b3;}
            QFrame#ButtonFrame { background-color: transparent; margin-top: 15px; border: none; }
            QPushButton { font-family: "Segoe UI Semibold", "Microsoft YaHei UI Semibold", sans-serif; font-size: 10.5pt; color: #FFFFFF; padding: 9px 18px; border-radius: 6px; border: none; min-height: 30px; text-align: center; }
            QPushButton#SimulateButtonPrimary { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #007BFF, stop:1 #0056b3); }
            QPushButton#SimulateButtonPrimary:hover { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0069D9, stop:1 #004085); }
            QPushButton#SimulateButtonPrimary:pressed { background-color: #004085; }
            QPushButton#ResetButton { background-color: #6C757D; color: #FFFFFF; }
            QPushButton#ResetButton:hover { background-color: #5A6268; }
            QPushButton#ResetButton:pressed { background-color: #495057; }
            QPushButton /* Export Button style */ { padding: 6px 12px; font-size: 9.5pt; min-height: 28px; background-color: #E9ECEF; color: #212529; border: 1px solid #CED4DA; }
            QPushButton:hover { background-color: #DEE2E6; border-color: #B9BEC3; }
            QPushButton:pressed { background-color: #D3D9DF; border-color: #AEB5BC; }
            QTextBrowser#ParamInfoDisplayCard, QTextBrowser#AnalysisDisplayCard { background-color: #FFFFFF; border: 1px solid #E0E5EA; border-radius: 8px; padding: 0px; }
            QWidget#RightPanel { background-color: #F4F6F8; } 
            QFrame#PlotCard { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E0E5EA; }
            QFrame#AnalysisSuperCard { border: none; background-color: transparent; }
            #MatplotlibToolbarModern { background-color: #FDFEFE; border-bottom: 1px solid #E0E5EA; padding: 2px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
            #MatplotlibToolbarModern QToolButton { background-color: transparent; border: 1px solid transparent; padding: 3px; margin: 1px; border-radius: 4px; }
            #MatplotlibToolbarModern QToolButton:hover { background-color: #E0E5EA; border: 1px solid #CCD1D9; }
            #MatplotlibToolbarModern QToolButton:pressed, #MatplotlibToolbarModern QToolButton:checked { background-color: #CCD1D9; border: 1px solid #A5B1C2; }
            QStatusBar#AppStatusBar { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.5pt; color: #495057; background-color: #E9ECEF; border-top: 1px solid #D1D8E0; padding-left: 10px; }
        """
        self.setStyleSheet(qss)

    @Slot()
    def reset_parameters(self):
        # ... (Same as your last provided version) ...
        for key, default_value in self.default_param_values.items():
            if key in self.param_inputs_widgets:
                widget = self.param_inputs_widgets[key]
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(default_value)
                elif isinstance(widget, QComboBox): widget.setCurrentIndex(default_value)
            slider_key = key + "_slider"
            if slider_key in self.param_inputs_widgets:
                slider_widget = self.param_inputs_widgets[slider_key]
                param_data = self.param_definitions.get(key)
                if param_data and len(param_data) == 5 and isinstance(param_data[-1], list):
                    slider_params = param_data[-1]; decimals = slider_params[3]; slider_multiplier = 10**decimals
                    slider_widget.setValue(int(default_value * slider_multiplier))
        self.status_bar.showMessage("参数已重置为默认值。", 4000)
        self.update_param_info_display("welcome")
        self.update_colormap_info_display(self.default_param_values["colormap"]) # Reset colormap info
        self.analysis_display.setHtml(self.analysis_display_initial_html)
        self.cdf_figure.clear(); self.cdf_canvas.draw()


    @Slot()
    def run_simulation(self):
        # ... (Same as your last provided version) ...
        self.status_bar.showMessage("正在初始化仿真..."); QApplication.processEvents()
        try:
            self.status_bar.showMessage("正在读取参数..."); QApplication.processEvents()
            bs_params = {'x': self.param_inputs_widgets["bs_x"].value(), 'y': self.param_inputs_widgets["bs_y"].value()}
            tx_power, antenna_gain, frequency = self.param_inputs_widgets["tx_power"].value(), self.param_inputs_widgets["antenna_gain"].value(), self.param_inputs_widgets["frequency"].value()
            env_type_key = self.param_inputs_widgets["env_type"].currentData()
            x_min, x_max, y_min, y_max, step = self.param_inputs_widgets["x_min"].value(), self.param_inputs_widgets["x_max"].value(), self.param_inputs_widgets["y_min"].value(), self.param_inputs_widgets["y_max"].value(), self.param_inputs_widgets["step"].value()
            if step <= 0: QMessageBox.warning(self, "参数错误", "网格步长必须为正数。"); self.status_bar.showMessage("仿真中止：步长错误。", 5000); return
            if frequency <= 0: QMessageBox.warning(self, "参数错误", "工作频率必须为正数。"); self.status_bar.showMessage("仿真中止：频率错误。", 5000); return
            if x_min > x_max or y_min > y_max: QMessageBox.warning(self, "范围错误", "模拟区域的最小范围不能大于最大范围。"); self.status_bar.showMessage("仿真中止：范围参数错误。", 5000); return
            elif x_min == x_max or y_min == y_max: QMessageBox.information(self, "范围提示", "模拟区域最小范围等于最大范围，将按点/线模拟。")
            self.status_bar.showMessage("正在创建网格..."); QApplication.processEvents()
            grid_xx, grid_yy = create_grid(x_min, x_max, y_min, y_max, step)
            if grid_xx.size == 0: QMessageBox.warning(self, "网格错误", "无法创建模拟网格。"); self.status_bar.showMessage("仿真中止：网格创建失败。", 5000); return
            self.status_bar.showMessage("正在计算信号强度..."); QApplication.processEvents()
            signal_strengths = simulate_signal_strength((grid_xx, grid_yy), bs_params, tx_power, frequency, antenna_gain, env_type_key)
            self.current_simulation_results = signal_strengths
            self.status_bar.showMessage("正在绘制覆盖图..."); QApplication.processEvents()
            self._plot_simulation_results(grid_xx, grid_yy, signal_strengths, bs_params, x_min, x_max, y_min, y_max)
            total_area_sqm = 0
            if grid_xx.ndim == 2 and grid_xx.shape[0] > 0 and grid_xx.shape[1] > 0:
                num_x_points, num_y_points = grid_xx.shape[1], grid_xx.shape[0]
                if num_x_points > 1 and num_y_points > 1: total_width = (num_x_points - 1) * (grid_xx[0,1] - grid_xx[0,0]); total_height = (num_y_points - 1) * (grid_yy[1,0] - grid_yy[0,0]); total_area_sqm = total_width * total_height
                else: total_area_sqm = step*step
            self.status_bar.showMessage("正在生成分析报告..."); QApplication.processEvents()
            self._generate_analysis_report(signal_strengths, total_area_sqm)
            self._plot_cdf_results(signal_strengths)
            self.status_bar.showMessage("仿真分析完成！结果已在右侧更新。", 5000)
        except ValueError as ve: QMessageBox.warning(self, "参数或计算错误", str(ve)); self.status_bar.showMessage(f"仿真失败：{ve}", 5000)
        except Exception as e: QMessageBox.critical(self, "执行错误", f"发生了一个意外错误: {e}"); self.status_bar.showMessage(f"仿真失败: {e}", 5000); import traceback; traceback.print_exc()


    def _plot_simulation_results(self, grid_xx, grid_yy, strengths, bs_params, x_min, x_max, y_min, y_max):
        # ... (Same as your last provided version) ...
        self.figure.clear(); ax = self.figure.add_subplot(111, facecolor='#FCFDFE')
        if strengths is None or strengths.size == 0: ax.text(0.5,0.5,"无有效数据显示", ha='center',va='center',transform=ax.transAxes,fontsize=14,color='gray'); self.canvas.draw(); return
        s_min_val, s_max_val = np.nanmin(strengths), np.nanmax(strengths)
        print(f"原始信号强度统计: Min={s_min_val:.2f}, Max={s_max_val:.2f}, Mean={np.nanmean(strengths):.2f}, Median={np.nanmedian(strengths):.2f}")
        selected_cmap_name = self.param_inputs_widgets["colormap"].currentText(); cmap = plt.get_cmap(self.cmaps.get(selected_cmap_name, 'viridis'))
        fixed_vmin, fixed_vmax = -125, -65; norm_vmin_actual, norm_vmax_actual = fixed_vmin, fixed_vmax
        if not (np.isnan(s_min_val) or np.isnan(s_max_val)):
            if s_min_val == s_max_val: norm_vmin_actual, norm_vmax_actual = s_min_val - 1, s_max_val + 1
        print(f"绘图使用的归一化范围: vmin={norm_vmin_actual:.2f}, vmax={norm_vmax_actual:.2f}"); norm = plt.Normalize(vmin=norm_vmin_actual, vmax=norm_vmax_actual)
        if np.all(np.isnan(strengths)): ax.text(0.5,0.5,"计算结果全部无效 (NaN)", ha='center',va='center',transform=ax.transAxes,fontsize=14,color='gray'); self.canvas.draw(); return
        step_val = self.param_inputs_widgets["step"].value() # Get current step for extent adjustment
        plot_extent = [x_min, x_max, y_min, y_max]
        if x_min == x_max: plot_extent[1] = x_max + step_val if step_val > 0 else x_max + 1
        if y_min == y_max: plot_extent[3] = y_max + step_val if step_val > 0 else y_max + 1
        if plot_extent[0] >= plot_extent[1]: plot_extent[1] = plot_extent[0] + 1
        if plot_extent[2] >= plot_extent[3]: plot_extent[3] = plot_extent[2] + 1
        im = ax.imshow(strengths, extent=plot_extent, origin='lower', cmap=cmap, aspect='auto', norm=norm, interpolation='bicubic')
        if not (np.isnan(s_min_val) or np.isnan(s_max_val)) and s_min_val != s_max_val and strengths.ndim == 2 and strengths.shape[0] > 1 and strengths.shape[1] > 1:
            contour_levels = [-115, -105, -95, -85, -75]; valid_contour_levels = [lvl for lvl in contour_levels if norm_vmin_actual < lvl < norm_vmax_actual and s_min_val < lvl < s_max_val]
            if valid_contour_levels:
                try: cs = ax.contour(grid_xx, grid_yy, strengths, levels=valid_contour_levels, colors='black', linewidths=0.7, alpha=0.8); ax.clabel(cs, inline=True, fontsize=7.5, fmt='%1.0f dBm', colors='black')
                except Exception as e_contour: print(f"绘制等高线时出错: {e_contour}")
        ax.scatter(bs_params['x'], bs_params['y'], color='#E74C3C', marker='h', s=200, edgecolor='black', linewidth=1.2, label='5G基站 (gNB)', zorder=10)
        ax.legend(fontsize=9.5, loc='upper right', frameon=True, facecolor='white', framealpha=0.85, edgecolor='#B0BEC5')
        ax.set_title('5G小区RSRP覆盖预测图', fontsize=14, weight='bold', color='#34495E'); ax.set_xlabel('X轴距离 (米)', fontsize=11, color='#4A5568'); ax.set_ylabel('Y轴距离 (米)', fontsize=11, color='#4A5568')
        ax.grid(True, linestyle=':', alpha=0.6, color='#BCCCDC', linewidth=0.6); ax.tick_params(axis='both', which='major', labelsize=9.5, colors='#5A6268'); ax.set_facecolor('#F8FAFC')
        try: cbar = self.figure.colorbar(im, ax=ax, shrink=0.9, aspect=25, pad=0.05, extend='both'); cbar.set_label('参考信号接收功率 RSRP (dBm)', fontsize=10.5, color='#4A5568'); cbar.ax.tick_params(labelsize=8.5, colors='#5A6268'); cbar.locator = MaxNLocator(nbins=7); cbar.update_ticks()
        except Exception as e_cbar: print(f"绘制颜色条时出错: {e_cbar}")
        self.figure.tight_layout(pad=1.8); self.canvas.draw()


    def _generate_analysis_report(self, strengths_data, total_area_sqm):
        # ... (Same as your last provided version, make sure html_global_css is defined or passed if needed) ...
        if strengths_data is None or strengths_data.size == 0 or np.all(np.isnan(strengths_data)): self.analysis_display.setHtml(self.analysis_display_initial_html); return
        max_rsrp, min_rsrp, avg_rsrp = np.nanmax(strengths_data), np.nanmin(strengths_data), np.nanmean(strengths_data)
        excellent_rsrp, good_rsrp, fair_rsrp, poor_rsrp = -80, -95, -105, -115
        cat_counts = {"excellent": np.sum(strengths_data >= excellent_rsrp), "good": np.sum((strengths_data >= good_rsrp) & (strengths_data < excellent_rsrp)), "fair": np.sum((strengths_data >= fair_rsrp) & (strengths_data < good_rsrp)), "marginal": np.sum((strengths_data >= poor_rsrp) & (strengths_data < fair_rsrp)), "poor": np.sum(strengths_data < poor_rsrp)}
        total_points = strengths_data.size if strengths_data.size > 0 else 1
        html_content = f"{self.html_global_css}<div class='card-header'><h3>覆盖分析与专业解读</h3></div><div class='card-content'><p><b>仿真区域概况：</b>...约 <b>{total_area_sqm:.0f} 平方米</b> ...</p><p><b>关键RSRP指标：</b><ul><li>最高RSRP: <span class='rsrp-excellent'>{max_rsrp:.1f} dBm</span></li><li>最低RSRP: <span class='rsrp-poor'>{min_rsrp:.1f} dBm</span></li><li>区域平均RSRP: {avg_rsrp:.1f} dBm</li></ul></p><h4>RSRP电平质量分布...：</h4><p>RSRP ...</p><table><tr><th>质量等级</th><th>RSRP范围 (dBm)</th><th>覆盖点占比</th><th>典型用户体验</th></tr>"
        levels_info = [("极好", f"≥ {excellent_rsrp}", cat_counts["excellent"], "rsrp-excellent", "高清视频流畅..."), ("良好", f"{good_rsrp} ~ {excellent_rsrp - 0.1:.1f}", cat_counts["good"], "rsrp-good", "网页浏览顺畅..."), ("一般", f"{fair_rsrp} ~ {good_rsrp - 0.1:.1f}", cat_counts["fair"], "rsrp-fair", "基本数据业务..."), ("边缘", f"{poor_rsrp} ~ {fair_rsrp - 0.1:.1f}", cat_counts["marginal"], "rsrp-marginal", "信号弱..."), ("差", f"< {poor_rsrp}", cat_counts["poor"], "rsrp-poor", "可能无法接入...")]
        for name, rsrp_range, count, style_class, experience in levels_info: percentage = (count / total_points) * 100 if total_points > 0 else 0; html_content += f"""<tr><td><span class='{style_class}'>{name}</span></td><td>{rsrp_range}</td><td>{percentage:.1f}% ({count}点)</td><td>{experience}</td></tr>"""
        html_content += "</table><div class='pro-tip'><b>专业解读：</b>...</div></div>"
        self.analysis_display.setHtml(html_content); self.analysis_display.moveCursor(QTextCursor.MoveOperation.Start)


    def _plot_cdf_results(self, strengths_data):
        # ... (Same as your last provided version) ...
        self.cdf_figure.clear()
        if strengths_data is None or strengths_data.size == 0 or np.all(np.isnan(strengths_data)): ax_cdf = self.cdf_figure.add_subplot(111); ax_cdf.text(0.5, 0.5, "无有效数据绘制CDF", ha='center', va='center', transform=ax_cdf.transAxes, color='gray'); self.cdf_canvas.draw(); return
        ax_cdf = self.cdf_figure.add_subplot(111, facecolor='#FCFDFE')
        data_flat = strengths_data.flatten(); data_flat = data_flat[~np.isnan(data_flat)]
        if data_flat.size == 0: ax_cdf.text(0.5, 0.5, "无有效数据绘制CDF", ha='center', va='center', transform=ax_cdf.transAxes, color='gray'); self.cdf_canvas.draw(); return
        sorted_data = np.sort(data_flat); yvals = np.arange(len(sorted_data)) / float(len(sorted_data) -1 if len(sorted_data) > 1 else 1)
        ax_cdf.plot(sorted_data, yvals, color='#007BFF', linewidth=1.8)
        ax_cdf.set_title('RSRP 累积分布函数 (CDF)', fontsize=11, weight='bold', color='#34495E'); ax_cdf.set_xlabel('RSRP (dBm)', fontsize=9, color='#4A5568'); ax_cdf.set_ylabel('概率 (P ≤ x)', fontsize=9, color='#4A5568')
        ax_cdf.grid(True, linestyle=':', alpha=0.7, color='#BCCCDC'); ax_cdf.tick_params(axis='both', which='major', labelsize=8, colors='#5A6268')
        rsrp_thresholds = {'差': -115, '边缘': -105, '一般': -95, '良好': -80}; data_min, data_max = sorted_data[0], sorted_data[-1]
        for label, threshold in rsrp_thresholds.items():
            if data_min < threshold < data_max:
                prob = np.sum(sorted_data <= threshold) / len(sorted_data)
                ax_cdf.axhline(y=prob, color='gray', linestyle='--', linewidth=0.6, alpha=0.7); ax_cdf.axvline(x=threshold, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)
                ax_cdf.text(threshold + 0.5, prob + 0.02, f'{threshold}dBm\n({prob*100:.0f}%)', fontsize=7, color='dimgray', ha='left', va='bottom')
        ax_cdf.set_ylim(0, 1.05); x_pad = (data_max - data_min) * 0.05 if (data_max - data_min) > 0 else 1; ax_cdf.set_xlim(data_min - x_pad, data_max + x_pad)
        self.cdf_figure.tight_layout(pad=0.8); self.cdf_canvas.draw()


    @Slot()
    def export_main_plot(self):
        # ... (Same as your last provided version) ...
        if self.current_simulation_results is None: QMessageBox.information(self, "无图可导", "请先运行一次仿真以生成覆盖图。"); return
        filePath, _ = QFileDialog.getSaveFileName(self, "保存覆盖图", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)")
        if filePath:
            try: self.figure.savefig(filePath, dpi=300, bbox_inches='tight'); self.status_bar.showMessage(f"覆盖图已保存至: {filePath}", 5000)
            except Exception as e: QMessageBox.critical(self, "导出失败", f"保存文件时出错: {e}"); self.status_bar.showMessage(f"导出覆盖图失败: {e}", 5000)


# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProfessionalSignalApp()
    window.show()
    sys.exit(app.exec())