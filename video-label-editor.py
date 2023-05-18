import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QObject
from PyQt5.QtWidgets import (QWidget, QComboBox, QPushButton, QLineEdit, QLabel, QCheckBox, QAction,
                             QTextEdit, QLCDNumber, QSlider, QListWidget, QAbstractItemView, QMessageBox,
                             QHBoxLayout, QFileDialog, QVBoxLayout, QApplication, QMainWindow, QGridLayout, QListWidgetItem)

import cv2
import os
import glob
import json

import numpy as np

import time

from opencv_frames import BboxFrame, Bbox, compute_iou

class AppWindow(QMainWindow):
    def __init__(self, screen_width, screen_height):
        super().__init__()       

        self.video_capture = None
        self.path_to_labelling_folder = None
        self.paths_to_labels_list = []
        self.path_to_video = None
        self.window_name = None
        self.frame_with_boxes = None
        self.img_rows = None
        self.img_cols = None

        self.screen_width = screen_width
        self.screen_height = screen_height



        self.autosave_mode = False

        # список с видимыми рамками. Это костыль, т.к. QListWidget почему-то не сохраняет выделенными строки
        self.temp_bboxes_list = []
     
        open_file_button = QPushButton("Open video")
        close_video_button = QPushButton("Close video")
        save_file_button = QPushButton("Save labels")
        next_frame_button = QPushButton("Next Frame")
        previous_frame_button = QPushButton("Previous Frame")
        self.autosave_current_checkbox = QCheckBox('Autosave Current Boxes')
        show_all_button = QPushButton('Show all classes')
        hide_all_button = QPushButton('Hide all classes')

        search_first_appearance_button = QPushButton("Search for first appearance")



        # чтение списка классов из json
        with open('settings.json', 'r', encoding='utf-8') as fd:
            self.settings_dict = json.load(fd)

        self.class_names_list = self.settings_dict['classes']
        
        #self.classes_combobox = QComboBox(self)
        
        #self.classes_combobox.addItems(self.class_names_list)

        # список отображаемых рамок
        self.visible_classes_list_widget = QListWidget()

        self.visible_classes_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        
        self.frame_display = QLCDNumber()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.reset_slider_display()
        self.frame_slider.valueChanged.connect(self.display_frame_position)

        # присоединение к обработчику события
        close_video_button.clicked.connect(self.close_video)
        open_file_button.clicked.connect(self.open_file)
        save_file_button.clicked.connect(self.save_labels_to_txt)
        next_frame_button.clicked.connect(self.next_frame_button_handling)
        previous_frame_button.clicked.connect(self.previous_frame_button_handling)
        self.autosave_current_checkbox.stateChanged.connect(self.autosave_current_checkbox_slot)

        self.visible_classes_list_widget.itemClicked.connect(self.update_visible_boxes_on_click_slot)
        self.visible_classes_list_widget.itemEntered.connect(self.update_visible_boxes_on_selection_slot)

        show_all_button.clicked.connect(self.show_all_button_slot)
        hide_all_button.clicked.connect(self.hide_all_button_slot)
        search_first_appearance_button.clicked.connect(self.search_first_appearance_button_slot)

        #self.classes_combobox.currentTextChanged.connect(self.update_current_box_class_name)

        # действия для строки меню
        open_file = QAction('Open', self)
        open_file.setShortcut('Ctrl+O')
        #openFile.setStatusTip('Open new File')
        open_file.triggered.connect(self.open_file)

        # строка меню
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(open_file)

        # выстраивание разметки приложения
        self.grid = QGridLayout()
        self.file_buttons_layout = QVBoxLayout()
        self.displaying_classes_layout = QVBoxLayout()
        self.horizontal_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()
        self.prev_next_layout = QHBoxLayout()

        self.prev_next_layout.addWidget(previous_frame_button)
        self.prev_next_layout.addWidget(next_frame_button)

        self.control_layout.addWidget(self.frame_display)
        self.control_layout.addWidget(self.frame_slider)
        #self.control_layout.addWidget(self.autosave_current_checkbox)
        self.control_layout.addLayout(self.prev_next_layout)

        self.file_buttons_layout.addWidget(open_file_button)
        self.file_buttons_layout.addWidget(close_video_button)
        self.file_buttons_layout.addWidget(save_file_button)

        # пока что спрячем разворачивающийся список классов...
        #self.displaying_classes_layout.addWidget(self.classes_combobox)
        self.displaying_classes_layout.addWidget(self.visible_classes_list_widget)
        self.displaying_classes_layout.addWidget(show_all_button)
        self.displaying_classes_layout.addWidget(hide_all_button)
        self.displaying_classes_layout.addWidget(search_first_appearance_button)

        #self.horizontal_layout.addLayout(self.file_buttons_layout)
        self.horizontal_layout.addLayout(self.control_layout)
        self.horizontal_layout.addLayout(self.displaying_classes_layout)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.horizontal_layout)
        self.setCentralWidget(self.main_widget)

        self.setWindowTitle('Video Label Editor')
        
        # Инициализируем поток для показа видео с подключением слотов к сигналам потока
        self.setup_imshow_thread()
        self.show()

    def show_info_message_box(self, window_title, info_text):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(window_title)
        msg_box.setText(info_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        return msg_box.exec()

    def search_first_appearance_button_slot(self):
        # Сначала надо проверить, что выделен лишь один класс
        qlist_len = self.visible_classes_list_widget.count()
        if qlist_len == 0:
            return

        selected_cnt = 0
        searching_class_name = None
        for item_idx in range(qlist_len):
            if self.visible_classes_list_widget.item(item_idx).isSelected():
                selected_cnt += 1
                searching_class_name = self.visible_classes_list_widget.item(item_idx).data(0)
            if selected_cnt > 1:
                self.show_info_message_box(window_title="Class search info", info_text="You should select only one class for searching")
                return

        if searching_class_name is None:
            self.show_info_message_box(window_title="Class search info", info_text="No class is selected")
            return
        
        for frame_idx, path in enumerate(self.paths_to_labels_list):
            with open(path, 'r') as fd:
                text = fd.read()
            if len(text) == 0:
                return
            
            text = text.split('\n')
            for str_bbox in text:
                try:
                    class_name, x0, y0, x1, y1 = str_bbox.split(',')
                except Exception:
                    continue
                if class_name == searching_class_name:
                    self.current_frame_idx = frame_idx
                    self.show_frame()
                    self.show_info_message_box(
                        window_title="Class search info",
                        info_text=f"First appearance of {class_name} at frame #{frame_idx}")
                    return
        self.show_info_message_box(
            window_title="Class search info",
            info_text=f"{searching_class_name} is not presented")

    def show_all_button_slot(self):
        self.show_or_hide(is_selected=True)
    
    def hide_all_button_slot(self):    
        self.show_or_hide(is_selected=False)

    def show_or_hide(self, is_selected):
        # определяем количество элементов в списке
        qlist_len = self.visible_classes_list_widget.count()
        for item_idx in range(qlist_len):
            class_name = self.visible_classes_list_widget.item(item_idx).data(0)
            for bbox in self.frame_with_boxes.bboxes_list:
                bbox_class_name = bbox.class_info_dict['class_name']
                if bbox_class_name == class_name:
                    bbox.is_visible = is_selected
                    self.visible_classes_list_widget.item(item_idx).setSelected(is_selected)
                    break


    def display_frame_position(self, current_frame_idx):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        self.frame_display.display(current_frame_idx)
        self.current_frame_idx = current_frame_idx
        self.show_frame()

    def autosave_current_checkbox_slot(self):
        '''
        Обработчик checkbox, отвечающего за автоматическое сохранение кадра при переходе на новый
        '''
        self.autosave_mode = self.autosave_current_checkbox.isChecked()
    
    def setup_imshow_thread(self):
        '''
        При инициализации нового потока необходимо также заново подключать все сигналы
        класса потока к соответствующим слотам главного потока
        '''
        self.imshow_thread = ImshowThread()
        self.imshow_thread.frame_update_signal.connect(self.update_visible_classes_list)


    def update_visible_boxes_on_click_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        class_str = item.data(0)
        class_name, bbox_idx = class_str.split(',')
        bbox_idx = int(bbox_idx)
        if self.frame_with_boxes is not None:
            for bbox in self.frame_with_boxes.bboxes_list:
                bbox_class_name = bbox.class_info_dict['class_name'] 
                sample_idx = bbox.class_info_dict['sample_idx'] 
                if bbox_class_name == class_name and bbox_idx == sample_idx:
                    bbox.is_visible = item.isSelected()

        self.update_visible_classes_list()

    
    def update_visible_boxes_on_selection_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        return

        class_str = item.data(0)
        class_name, bbox_idx = class_str.split(',')
        bbox_idx = int(bbox_idx)
        if self.frame_with_boxes is not None:
            for bbox in self.frame_with_boxes.bboxes_list:
                bbox_class_name = bbox.class_info_dict['class_name'] 
                sample_idx = bbox.class_info_dict['sample_idx'] 
                if bbox_class_name == class_name and bbox_idx == sample_idx:
                    bbox.is_visible = item.isSelected()


    def load_labels_from_txt(self):
        '''
        Загружаем из txt-файлов координаты рамок и информацию о классах. 
        Информация загружается в self.frame_with_boxes, 
        self.visible_classes_list_widget не изменяется
        '''
        path_to_to_loading_labels = os.path.join(self.path_to_labelling_folder, '{:07d}.txt'.format(self.current_frame_idx))
        if os.path.isfile(path_to_to_loading_labels):
            # сохраняем список рамок предыдущего кадра
            self.temp_bboxes_list = self.frame_with_boxes.bboxes_list
            
            with open(path_to_to_loading_labels, 'r') as fd:
                text = fd.read()
            if len(text) == 0:
                return
            
            text = text.split('\n')

            # словарь нужен для того, чтобы подсчитывать количество рамок одного класса
            new_bboxes_dict = {}
            for str_bbox in text:
                try:
                    class_name, x0, y0, x1, y1 = str_bbox.split(',')
                except Exception:
                    continue
                if self.img_cols / self.screen_width > 0.65 or self.img_rows / self.screen_height > 0.65:
                    scaling_factor = 0.65*self.screen_width/self.img_cols
                    scaling_function = lambda x: int(scaling_factor*int(x))
                else:
                    scaling_function = int
                # !!!
                #scaling_factor = self.img_cols*1.5/0.7/self.screen_width
                #scaling_function = lambda x: int(scaling_factor*int(x))
                x0, y0, x1, y1 = tuple(map(scaling_function, (x0, y0, x1, y1)))
                color = self.frame_with_boxes.palette_dict[class_name]
                try:
                    new_bboxes_dict[class_name].append((color, x0, y0, x1, y1))
                except KeyError:
                    new_bboxes_dict[class_name] = [(color, x0, y0, x1, y1)]
            
            # создаем новый список рамок
            new_bboxes_list = []
            for class_name, coords_list in new_bboxes_dict.items():
                for bbox_idx, (color, x0, y0, x1, y1) in enumerate(coords_list):
                    is_visible = False
                    for prev_bbox in self.temp_bboxes_list:
                        prev_class_name, prev_color, prev_bbox_idx = tuple(prev_bbox.class_info_dict.values())
                        
                        if bbox_idx == prev_bbox_idx and class_name == prev_class_name:
                            is_visible = prev_bbox.is_visible
                            break

                    new_bbox = Bbox(x0, y0, x1, y1, self.img_cols, self.img_rows, class_name, color, bbox_idx, is_visible)
                    new_bboxes_list.append(new_bbox)

            # переиндексируем рамки

            for class_name, bboxes_list in new_bboxes_dict.items():
                self.frame_with_boxes.class_indices_dict[class_name] = len(bboxes_list)
            if len(new_bboxes_list) != 0:

                self.frame_with_boxes.bboxes_list = new_bboxes_list
            


    #@pyqtSlot()
    def update_visible_classes_list(self):
        '''
        обновление списка рамок. 
        Информация о рамках берется из списка рамок, хранящегося в self.frame_with_boxes
        '''

        # определяем количество элементов в списке
        qlist_len = self.visible_classes_list_widget.count()


        new_list = []
        for bbox_idx, bbox in enumerate(self.frame_with_boxes.bboxes_list):
            class_name = bbox.class_info_dict['class_name']
            sample_idx = bbox.class_info_dict['sample_idx']
            is_selected = bbox.is_visible

            displayed_name = f'{class_name},{sample_idx}'
            item = QListWidgetItem(displayed_name)
            
            for item_idx in range(qlist_len):
                prev_item = self.visible_classes_list_widget.item(item_idx)

                prev_data = prev_item.data(0)
                if prev_data == displayed_name:
                    item = prev_item
                    break

            new_list.append({'data': item.data(0), 'is_selected': is_selected})

        
        # обновление списка классов?
        self.visible_classes_list_widget.clear()
        for bbox_idx, data_dict in enumerate(new_list):
            item = QListWidgetItem(data_dict['data'])
            self.visible_classes_list_widget.addItem(item)
            self.visible_classes_list_widget.item(bbox_idx).setSelected(data_dict['is_selected'])
        
            
    def update_current_box_class_name(self, class_name):
        if self.frame_with_boxes is not None:
            self.frame_with_boxes.update_current_class_name(class_name)

    def reset_slider_display(self):
        '''
        Обнуление значений на экране и на слайдере
        '''
        self.frame_slider.setRange(0, 0)
        self.set_slider_display_value(0)

    def set_slider_display_value(self, val):
        self.frame_slider.setValue(val)
        self.frame_display.display(val)

    def setup_slider_range(self, max_val, current_idx):
        '''
        Установка диапазона значений слайдера
        '''
        self.frame_slider.setRange(0, max_val)
        self.set_slider_display_value(current_idx)
        

    def close_video(self):
        '''
        Обработчик закрытия файла
        Сохраняет рамки, закрывает поток чтения изображения и делает объект изображения с рамками пустым
        '''

        if self.frame_with_boxes is not None:
            self.save_labels_to_txt()
        self.close_imshow_thread()
        self.frame_with_boxes = None
        self.reset_slider_display()

        
    def open_file(self):
        self.close_imshow_thread()
        # получаем абсолютный путь до файла
        title = 'Open video'

        # записываем в файл settings.json путь до папки с последним открытым файлом
        try:
            last_opened_folder_path = self.settings_dict['last_opened_folder']
        except KeyError:
            last_opened_folder_path = '/home'

        
        # фильтр разрешений файлов
        file_filter = 'Videos (*.mp4 *.wmw *.avi *.mpeg)'
        open_status_tuple = QFileDialog.getOpenFileName(self, title, last_opened_folder_path, file_filter)
        path = open_status_tuple[0]
        if len(path) == 0:
            return

        path = os.sep.join(path.split('/'))
        path_to_folder, name = os.path.split(path)   

        self.settings_dict['last_opened_folder'] = path_to_folder
        with open('settings.json', 'w', encoding='utf-8') as fd:
            json.dump(self.settings_dict, fd)

        label_folder_name = '.'.join(name.split('.')[:-1]) + '_labels'
        
        self.path_to_labelling_folder = os.path.join(path_to_folder, label_folder_name)

        if os.path.isdir(self.path_to_labelling_folder):
            # список путей до txt файлов с координатами рамок номер кадра совпадает с именем файла
            self.paths_to_labels_list = glob.glob(os.path.join(self.path_to_labelling_folder, '*.txt'))
        else:
            self.paths_to_labels_list = []
            os.mkdir(self.path_to_labelling_folder)

        self.video_capture = cv2.VideoCapture(path)
        ret, frame = self.video_capture.read()
        if not ret:
            raise RuntimeError(f'Can not read {path} video')
        
        self.frame_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_rows, self.img_cols = frame.shape[:2]

        if len(self.paths_to_labels_list) > 0:
            self.current_frame_idx = len(self.paths_to_labels_list) - 1
        else:
            self.current_frame_idx = 0
        
        self.current_frame_idx = 0

        self.setup_slider_range(max_val=self.frame_number, current_idx=self.current_frame_idx)

        self.window_name = name
        
        self.frame_with_boxes = BboxFrame(frame, self.class_names_list, self.class_names_list[0])
        self.setup_imshow_thread()
        
        self.imshow_thread.setup_frame(self.frame_with_boxes, self.window_name)
        
        self.imshow_thread.start()

        # сразу открываем видео
        self.show_frame()

    def close_imshow_thread(self):
        if self.imshow_thread.isRunning():
            self.frame_with_boxes.delete_img()            
            self.imshow_thread.wait()

    def save_labels_to_txt(self):
        '''
        Сохранение координат рамок и классов в txt-файл, имя которого совпадает с номером кадра
        СОХРАНЕНИЕ ВЫПОЛНЯЕТСЯ АВТОМАТИЧЕСКИ ПРИ ПЕРЕХОДЕ НА СЛЕДУЮЩИЙ КАДР.
        '''
        if self.frame_with_boxes is not None:
            bboxes = []
            for bbox in self.frame_with_boxes.bboxes_list:
                x0,y0,x1,y1 = bbox.coords
                class_name = bbox.class_info_dict['class_name']
                bboxes.append(f'{class_name},{x0},{y0},{x1},{y1}')

            bboxes = '\n'.join(bboxes)

            path_to_to_saving_labels = os.path.join(self.path_to_labelling_folder, '{:07d}.txt'.format(self.current_frame_idx))
            with open(path_to_to_saving_labels, 'w') as fd:
                fd.write(bboxes)


    def previous_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return

        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            return
        self.show_frame()


    def next_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        # сохраняем все рамки
        if self.current_frame_idx > -1:
            if self.autosave_mode:
                self.save_labels_to_txt()
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.frame_number:
            return
        self.show_frame()


    def show_frame(self):
        if self.video_capture is None or self.current_frame_idx >= self.frame_number:
            return
        if self.current_frame_idx < 0:
            self.current_frame_idx = 0
            return
        
        self.set_slider_display_value(self.current_frame_idx)
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        ret, frame = self.video_capture.read()
        if self.img_cols / self.screen_width > 0.65 or self.img_rows / self.screen_height > 0.65:
            scaling_factor = 0.65*self.screen_width/self.img_cols
            new_size = tuple(map(lambda x: int(scaling_factor*x), (self.img_cols, self.img_rows)))
            frame = cv2.resize(frame, new_size)
        
        if ret:
            self.frame_with_boxes.update_img(frame)
            
            self.load_labels_from_txt()
            self.update_visible_classes_list()
            

    def stop_showing(self):
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()


class BoxesCheckingWindow(AppWindow):
    def update_visible_classes_list(self):
        '''
        Поведение отличается от родительского тем, что здесь мы отображаем в принципе все классы,
        независимо от того, есть они в кадре или нет.
        '''
        
        # заполняем список всех возможных классов, если он пуст
        qlist_len = self.visible_classes_list_widget.count()
        if qlist_len == 0:
            for bbox_idx, class_name in enumerate(self.class_names_list):
                displayed_name = f'{class_name}'
                item = QListWidgetItem(displayed_name)
                self.visible_classes_list_widget.addItem(item)
            qlist_len = self.visible_classes_list_widget.count()

        for item_idx in range(qlist_len):
            item = self.visible_classes_list_widget.item(item_idx)
            is_selected = self.visible_classes_list_widget.item(item_idx).isSelected()
            class_name = item.data(0)
            for bbox in self.frame_with_boxes.bboxes_list:
                actual_class_name = bbox.class_info_dict['class_name']
                #sample_idx = bbox.class_info_dict['sample_idx']
                #is_selected = bbox.is_visible
                if class_name==actual_class_name:
                    if is_selected:
                        #self.visible_classes_list_widget.item(item_idx).setSelected(True)
                        bbox.is_visible = True
                    break

    def update_visible_boxes_on_click_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        class_name = item.data(0)
        #class_name, bbox_idx = class_str.split(',')
        #bbox_idx = int(bbox_idx)
        if self.frame_with_boxes is not None:
            for bbox in self.frame_with_boxes.bboxes_list:
                bbox_class_name = bbox.class_info_dict['class_name'] 
                #sample_idx = bbox.class_info_dict['sample_idx'] 
                if bbox_class_name == class_name:# and bbox_idx == sample_idx:
                    bbox.is_visible = item.isSelected()

        self.update_visible_classes_list()


class ImshowThread(QThread):
    frame_update_signal = pyqtSignal()

    def  __init__(self, parent=None):
        super().__init__(parent)
        self.frame_with_boxes = None
        self.window_name = None
        self.is_showing = False
        self.is_drawing = False
    
    def setup_frame(self, frame_with_boxes, window_name):
        self.frame_with_boxes = frame_with_boxes
        self.window_name = window_name

    def run(self):
        self.init_showing_window()
        # почему-то работает только это условие...
        # потом надо переписать,
        while self.frame_with_boxes.img is not None:
            if self.frame_with_boxes.is_bboxes_changed:
                self.frame_update_signal.emit()
                self.frame_with_boxes.is_bboxes_changed = False

            img_with_boxes = self.frame_with_boxes.render_boxes()
            cv2.imshow(self.window_name, img_with_boxes)
            key = cv2.waitKey(20)

        self.frame_with_boxes = None
        self.stop_showing()

    def init_showing_window(self):
        if not self.is_showing:
            self.is_showing = True
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.frame_with_boxes)

    def stop_showing(self):
        
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    ex = BoxesCheckingWindow(screen_resolution.width(), screen_resolution.height())
    
    sys.exit(app.exec_())