import numpy as np
import cv2

from PIL import Image, ImageFont, ImageDraw, ImageColor, ImageFont

import pickle

def create_palette(classes_list):
    max_color = 0xFFFFFF
    color_step = max_color // len(classes_list)

    color = 0
    palette = {}
    for person in classes_list:
        # преобразуем целочисленное выражение цвета в RGB кортеж
        r, g, b = ImageColor.getrgb('#{:06x}'.format(color)) # 06x - формат шестиразрядного шестнадцатиричного числа с заполнением нулями пустых разрядов
        palette[person] = (r, g, b)
        color += color_step
    return palette

def check_cursor_in_bbox(x0, y0, x1, y1, cursor_x, cursor_y):
    if cursor_x > x0 and cursor_y > y0 and cursor_x < x1 and cursor_y < y1:
        return True
    return False

def check_cursor_in_corner(corner_x, corner_y, cursor_x, cursor_y, target_radius):
    radius = np.linalg.norm([cursor_x-corner_x, cursor_y-corner_y])
    if radius <= target_radius:
        return True
    return False

def process_box_coords(x0, y0, x1, y1, rows, cols):
    # превращаем строки в числа и фиксируем координаты рамки, чтобы они не выходили за пределы кадра
    x0 = np.clip(int(x0), 0, cols)
    x1 = np.clip(int(x1), 0, cols)
    y0 = np.clip(int(y0), 0, rows)
    y1 = np.clip(int(y1), 0, rows)

    # чтобы у нас ширина и высота рамки была не отрицательной, 
    # переставляем местами нулевую и первую координаты, если первая больше нулевой
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    return x0, y0, x1, y1

def xyxy2xywh(x0, y0, x1, y1):
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    w = max(x0, x1) - min(x0, x1)
    h = max(y0, y1) - min(y0, y1)
    return x0, y0, w, h

def compute_iou(coords1, coords2):
    x00, y00, x01, y01 = coords1
    x10, y10, x11, y11 = coords2
    
    x0 = abs(x01 - x00)
    y0 = abs(y01 - y00)

    x1 = abs(x11 - x10)
    y1 = abs(y11 - y10)

    # вычисление площадей отдельных прямоугольников
    s0 = x0*y0
    s1 = x1*y1
    
    # вычисление координат пересечения 
    # если рамки перескаются, то строка или столбец
    # левого верхнего угла одной из рамок должен быть 
    # меньше строки или слобца левого верхнего угла другой рамки соответственно. 
    # Строка или столбец правого нижнего угла одной из рамок должна 
    # быть больше или равна строке или столбцу правого нижнего угла другой рамки соответственно
    x0 = max(x00, x10) 
    y0 = max(y00, y10)
    x1 = min(x01, x11)
    y1 = min(y01, y11)

    # если хотя бы одна из сторон оказалась меньше нуля, значит пересечения нет
    if x1-x0 <= 0 or y1-y0 <= 0:
        s_intersection = 0
    else: 
        s_intersection = (x1-x0)*(y1-y0)

    return s_intersection / (s0 + s1 - s_intersection + 1e-9)

    
def draw_bbox_with_text(
    image:np.array,
    bbox_coords:tuple,
    bbox_width:int,
    class_name:str,
    color:tuple,
    font:ImageFont.FreeTypeFont,
    ):
    '''
    image:np.array - массив пикселей изображения
    bbox_coords:tuple|list, - координаты рамки в формате x0,y0,x1,y1
    bbox_width:int, - ширина рамки
    class_name:str, - имя выводимого класса
    color:tuple|list, - цвет рамки
    font:ImageFont.FreeTypeFont, - шрифт текста
    '''

    x0, y0, x1, y1 = bbox_coords
    image =  Image.fromarray(image)
    cols, rows = image.size

    draw = ImageDraw.Draw(image)

    # рисуем прямоугольник для общей рамки...
    draw.rectangle(bbox_coords, outline=color, width=bbox_width)
   

    # определяем цвет шрифта исходя из яркости ЧБ эквивалента цвета класса
    r, g, b = color
    grayscale = int(0.299*r + 0.587*g + 0.114*b)
    # пороговая фильтрация работает на удивление хорошо...
    font_color = 255 if grayscale < 128 else 0

    # вычисляем координаты текста - посередине рамки
    text_coords = ((x1+x0)//2, (y1+y0)//2)
    #print(text_coords)

    font_size = font.size

    # квадратный корень почему-то работает очень хорошо для вычисления ширины рамки текста...
    text_bbow_width = np.round(np.sqrt(font_size)).astype(int)
    # вычисляем зазор между рамкой текста и текстом
    text_bbox_spacing = text_bbow_width//3 if text_bbow_width//3 > 1 else 1

    # определяем координаты обрамляющего текст прямоугольника
    text_bbox = draw.textbbox(text_coords, class_name, font=font, anchor='mm') # anchor='mm' означает расположение текста посередине относительно координат
    # расширяем рамку на 3 пикселя в каждом направлении
    text_bbox = tuple(np.add(text_bbox, (-text_bbow_width, -text_bbow_width, text_bbow_width, text_bbow_width)))

    # рисуем прямоугольник для текста
    draw.rectangle(text_bbox, outline=(font_color, font_color, font_color), fill=color, width=bbox_width-text_bbox_spacing)
    # пишем текст
    draw.text(text_coords, class_name, font=font, anchor='mm', fill=(font_color, font_color, font_color))

    #print(cols, rows)
    return np.array(image)


class BboxFrame:
    def __init__(self, img, class_names_list, current_class_name):
        '''
        Задаем изображение, список классов и имя класса, актуальной рамки
        '''
        self.img = img.copy()

        # список, где мы храним все рамки
        self.bboxes_list = []

        # имена всех классов
        self.class_names_list = class_names_list

        # словарь, содержащий цвета для классов рамок
        self.palette_dict = create_palette(class_names_list)

        # словарь для индексирования рамок различны[ классов]
        self.class_indices_dict = {class_name: 0 for class_name in class_names_list}

        # координаты изменяемого угла
        self.displayed_corner = None
        # координаты изменяемой рамки 
        self.displayed_box = None 

        # рамка, которая в данный момент или создается, или изменяется
        self.processing_box = None

        # имя класса конкретной рамки
        self.current_class_name = current_class_name

        self.delete_box_flag = False

        # флаг, сигнализирующий о том, что рамки каким-то образом изменились
        self.is_bboxes_changed = False

        # флаг, сигнализирующий о том, что мы показываем номер рамки одного и того же класса
        self.is_bbox_idx_displayed = True
    
    def update_palette(self, new_classes_list):
        self.class_names_list = new_classes_list

        self.palette_dict = create_palette(new_classes_list)

    def draw_one_box(self, event, flags, x, y):
        rows, cols, channels = self.img.shape
        
        if event == cv2.EVENT_LBUTTONDOWN and not flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.processing_box is None:
                # получаем индекс текущего класса
                class_idx = self.class_indices_dict[self.current_class_name]
                color = self.palette_dict[self.current_class_name]                
                self.processing_box = Bbox(x, y, x, y, rows, cols, self.current_class_name, color, class_idx)
                self.processing_box.create_bbox(x, y)
                self.bboxes_list.append(self.processing_box)
                # записываем индекс рамки
                self.class_indices_dict[self.current_class_name] += 1

        # mouse is being moved, draw rectangle
        elif event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.processing_box is not None:
                if self.processing_box.is_bbox_creation:
                    # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                    self.bboxes_list.pop()
                    self.processing_box.create_bbox(x, y)
                    self.bboxes_list.append(self.processing_box)
                    
        # if the left mouse button was released, set the drawing flag to False
        elif event == cv2.EVENT_LBUTTONUP and not flags & cv2.EVENT_FLAG_CTRLKEY:
            # фиксируеми нарисованную рамку
            if self.processing_box is not None and self.processing_box.is_bbox_creation:
                self.bboxes_list.pop()
                self.processing_box.create_bbox(x, y)
                
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_bbox_creation()

                self.bboxes_list.append(self.processing_box)

                self.processing_box = None
                
                self.is_bboxes_changed = True

                
        else:
            self.is_bboxes_changed = False


    def correct_rectangle(self, event, flags, bbox_idx, x, y):
        rows, cols, channels = self.img.shape
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is None:
                    # здесь self.processing_box должен быть проинициализирован, иначе возвращаемся из функции
                    self.processing_box = self.bboxes_list.pop(bbox_idx)
                    self.processing_box.corner_drag(x, y)
                    self.bboxes_list.append(self.processing_box)
        # mouse is being moved, draw rectangle
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None:
                    if self.processing_box.is_corner_dragging:
                        # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                        self.bboxes_list.pop()
                        self.processing_box.corner_drag(x, y)
                        self.bboxes_list.append(self.processing_box)

            else:
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_corner_drag()
                self.processing_box = None
                    
        # if the left mouse button was released, set the drawing flag to False
        elif event == cv2.EVENT_LBUTTONUP:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None and self.processing_box.is_corner_dragging:
                    # фиксируеми нарисованную рамку
                    self.processing_box.is_corner_dragging = False
                    self.bboxes_list.pop()
                    self.processing_box.corner_drag(x, y)
                    self.processing_box.make_x0y0_lesser_x1y1()
                    self.processing_box.stop_corner_drag()
                
                    self.bboxes_list.append(self.processing_box)

                    self.processing_box = None

                    self.is_bboxes_changed = True
        else:
            self.processing_box = None
            self.is_bboxes_changed = False

    def drag_box(self, event, flags, bbox_idx, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is None:
                    # извлекаем корректируемую рамку из списка
                    self.processing_box = self.bboxes_list.pop(bbox_idx)
                    self.processing_box.box_drag(x, y)
                    self.bboxes_list.append(self.processing_box)
        # mouse is being moved, draw rectangle
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None:
                    if self.processing_box.is_bbox_dragging:
                        # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                        self.bboxes_list.pop()
                        self.processing_box.box_drag(x, y)
                        self.bboxes_list.append(self.processing_box)
            else:
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_box_drag()
                self.processing_box = None                
        # if the left mouse button was released, set the drawing flag to False
        elif event == cv2.EVENT_LBUTTONUP:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None:
                    if self.processing_box.is_bbox_dragging:
                        self.bboxes_list.pop()
                        self.processing_box.box_drag(x, y)
                        self.processing_box.make_x0y0_lesser_x1y1()
                        self.processing_box.stop_box_drag()

                        self.bboxes_list.append(self.processing_box)

                        self.processing_box = None
                        self.is_bboxes_changed = True
        else:
            self.is_bboxes_changed = False
    
    
    def change_class_name(self, event, flags, bbox_idx):
        if event == cv2.EVENT_RBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY and not flags & cv2.EVENT_FLAG_ALTKEY:
                if self.processing_box is None:
                    self.processing_box = self.bboxes_list.pop(bbox_idx)
                    current_color = self.palette_dict[self.current_class_name]
                    self.processing_box.update_class_info(self.current_class_name, current_color)
                    self.bboxes_list.append(self.processing_box)
                    self.processing_box = None
                    self.is_bboxes_changed = True
        else:
            self.is_bboxes_changed = False



    def delete_box(self, event, flags, bbox_idx):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_ALTKEY and not flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is None:
                    self.processing_box = self.bboxes_list.pop(bbox_idx)
                    
                    self.processing_box = None
                    self.is_bboxes_changed = True
        else:
            self.is_bboxes_changed = False

    def update_img(self, img):
        self.img = img.copy()

    def delete_img(self):
        self.img = None

    def update_current_class_name(self, current_class_name):
        self.current_class_name = current_class_name

    def __call__(self, event, x, y, flags, param):
        '''
        Обработка коллбэков
        '''
        if flags & cv2.EVENT_FLAG_CTRLKEY and not flags & cv2.EVENT_FLAG_ALTKEY:
            self.delete_box_flag = False
            if self.processing_box is not None:
                self.displayed_box = self.processing_box.coords
                if self.processing_box.is_corner_dragging:
                    self.correct_rectangle(event, flags, -1, x, y)
                elif self.processing_box.is_bbox_dragging:
                    self.drag_box(event, flags, -1, x, y)   
            else:
                for bbox_idx, bbox in enumerate(self.bboxes_list):
                    x0, y0, x1, y1 = bbox.coords
                    
                    if check_cursor_in_corner(x0,y0,x,y,6):
                        self.displayed_corner = (x0, y0)
                        self.displayed_box = None
                        
                        self.correct_rectangle(event, flags, bbox_idx, x, y)
                        break
                        
                    elif check_cursor_in_corner(x1,y1,x,y,6):
                        self.displayed_corner = (x1, y1)
                        self.displayed_box = None
                        
                        self.correct_rectangle(event, flags, bbox_idx, x, y)
                        break
                    elif check_cursor_in_bbox(x0, y0, x1, y1, x, y):
                        self.displayed_box = (x0, y0, x1, y1)
                        self.displayed_corner = None
                        self.drag_box(event, flags, bbox_idx, x, y)
                        self.change_class_name(event, flags, bbox_idx)
                        break
                    else:
                        self.displayed_corner = None
                        self.displayed_box = None
                        self.is_bboxes_changed = False
        
        elif flags & cv2.EVENT_FLAG_ALTKEY and not flags & cv2.EVENT_FLAG_CTRLKEY:
            for bbox_idx, bbox in enumerate(self.bboxes_list):
                x0, y0, x1, y1 = bbox.coords
                if check_cursor_in_bbox(x0, y0, x1, y1, x, y):
                    self.displayed_box = (x0, y0, x1, y1)
                    self.displayed_corner = None
                    self.delete_box_flag = True
                    self.delete_box(event, flags, bbox_idx)
                else:
                    self.displayed_box = None
                    self.delete_box_flag = False
                    self.is_bboxes_changed = False
        else:
            self.draw_one_box(event, flags, x, y)

    def update_bboxes_list(self, new_bboxes_list):
        self.bboxes_list = new_bboxes_list

    def render_boxes(self):        
        drawing_img = self.img.copy()
        rows, cols, channels = drawing_img.shape
        # определяем размер шрифта исходя из размера изображения
        font_size = min(rows,cols)//30
        # устанавливаем шрифт для указания размечаемых людей
        font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        # вычисляем ширину рамки. Квадратный корень почему-то работает хорошо...
        line_width = np.round(np.sqrt(font_size).astype(int))
        for idx, bbox in enumerate(self.bboxes_list):
            if bbox.is_visible:
                x0, y0, x1, y1 = bbox.coords
                class_name = bbox.class_info_dict['class_name']
                sample_idx = bbox.class_info_dict['sample_idx']
                if self.is_bbox_idx_displayed:
                    displaying_name = f'{class_name},{sample_idx}'
                else:
                    displaying_name = class_name
                color = bbox.class_info_dict['color']
                drawing_img = draw_bbox_with_text(drawing_img, (x0,y0,x1,y1), line_width, displaying_name, color, font)
                drawing_img = cv2.circle(drawing_img, (x0, y0), 6, color, -1)
                if bbox.is_bbox_creation:
                    drawing_img = cv2.circle(drawing_img, (x1, y1), 6, (0, 0, 255), -1)
                elif bbox.is_corner_dragging:
                    if (bbox.ix, bbox.iy) == (x0, y0):
                        drawing_img = cv2.circle(drawing_img, (x1, y1), 6, (0, 0, 255), -1)
                    elif (bbox.ix, bbox.iy) == (x1, y1):
                        drawing_img = cv2.circle(drawing_img, (x0, y0), 6, (0, 0, 255), -1)
                else:
                    drawing_img = cv2.circle(drawing_img, (x1, y1), 6, color, -1)

                if self.displayed_corner is not None and not bbox.is_corner_dragging:
                    drawing_img = cv2.circle(drawing_img, self.displayed_corner, 6, (0, 0, 255), -1)

                if self.displayed_box is not None:
                    x0,y0,x1,y1 = self.displayed_box
                    if self.delete_box_flag:
                        thickness = -1
                    else:
                        thickness = 4
                    
                    drawing_img = cv2.rectangle(drawing_img, (x0, y0), (x1, y1), (0, 0, 255), thickness)
        return drawing_img


class Bbox:
    def __init__(self, x0, y0, x1, y1, img_rows, img_cols, class_name, color, sample_idx, is_visible=True, is_manually_manipulated=True):
        '''
        x0, y0, x1, y1 - координаты правого верхнего и левого нижнего углов рамки
        img_rows, img_cols - количество строк и стоблцов изображения, необходимые для нормировки координат рамок
        class_name - имя класса
        color - цвет рамки
        sample_idx - индекс или номер рамки одного и того же класса для отслеживания ситуаций, когда на изображении много объектов одного и того же класса 
        is_manually_manipulated - флаг-сигнал ручного создания/изменения рамки. Нужен для отличения созданных вручную и автоматически сгенерированных рамок
        '''
        self.coords = (x0, y0, x1, y1)
        self.class_name = class_name
        self.class_info_dict = {'class_name': class_name, 'color': color, 'sample_idx': sample_idx}

        # координаты начального угла рамки
        self.ix = None
        self.iy = None

        self.dx0 = None
        self.dy0 = None
        self.dx1 = None
        self.dy1 = None

        self.is_bbox_creation = False
        self.is_corner_dragging = False
        self.is_bbox_dragging = False

        self.img_rows = img_rows
        self.img_cols = img_cols

        # флаг для отображения и для трекинга
        self.is_visible = is_visible

        # флаг-сигнал ручного создания/изменения рамки. Нужен для отличения созданных вручную и автоматически сгенерированных рамок
        self.is_manually_manipulated = is_manually_manipulated

    def x0y0x1y1_to_x0y0wh(self):
        '''
        Перевод координат из формата x0y0x1y1 в формат x0,y0, ширина, высота
        для обеспечения работы трекера opencv
        '''
        x0,y0,x1,y1 = self.coords
        return x0,y0,x1-x0,y1-y0

    def corner_drag(self, corner_x, corner_y):
        x0, y0, x1, y1 = self.coords
        self.is_bbox_creation = False
        self.is_bbox_dragging = False
        if not self.is_corner_dragging:
            dist0 = np.linalg.norm([x0-corner_x, y0-corner_y])
            dist1 = np.linalg.norm([x1-corner_x, y1-corner_y])
            if dist0 >= dist1:
                self.ix = x0
                self.iy = y0
            else:
                self.ix = x1
                self.iy = y1
            self.is_corner_dragging = True
        self.update_coords(self.ix, self.iy, corner_x, corner_y)

    def create_bbox(self, corner_x, corner_y):
        x0, y0, x1, y1 = self.coords
        self.is_bbox_dragging = False
        self.is_corner_dragging = False
        if not self.is_bbox_creation:
            dist0 = np.linalg.norm([x0-corner_x, y0-corner_y])
            dist1 = np.linalg.norm([x1-corner_x, y1-corner_y])
            if dist0 <= dist1:
                self.ix = x0
                self.iy = y0
            else:
                self.ix = x1
                self.iy = y1
            self.is_bbox_creation = True
        self.update_coords(self.ix, self.iy, corner_x, corner_y)

    def compute_initial_corner(self, x0, y0, x1, y1, corner_x, corner_y):
        # определяем, к какому углу ближе курсор
        dist0 = np.linalg.norm([x0-corner_x, y0-corner_y])
        dist1 = np.linalg.norm([x1-corner_x, y1-corner_y])
        if dist0 <= dist1:
            self.ix = x0
            self.iy = y0
        else:
            self.ix = x1
            self.iy = y1

    def box_drag(self, x, y):
        self.is_corner_dragging = False
        self.is_bbox_creation = False
        if not self.is_bbox_dragging:
            x0, y0, x1, y1 = self.coords
            self.dx0 = x - x0
            self.dy0 = y - y0

            self.dx1 = x1 - x
            self.dy1 = y1 - y

            self.is_bbox_dragging = True
        else:
            x0 = x - self.dx0
            y0 = y - self.dy0

            x1 = self.dx1 + x
            y1 = self.dy1 + y
            self.update_coords(x0, y0, x1, y1)

    def stop_corner_drag(self):
        self.is_corner_dragging = False
        self.ix = None
        self.iy = None

    def stop_bbox_creation(self):
        self.is_bbox_creation = False
        self.ix = None
        self.iy = None

    def stop_box_drag(self):
        self.is_bbox_dragging = False
        self.dx0 = None
        self.dy0 = None
        self.dx1 = None
        self.dy1 = None

        
    def update_coords(self, x0, y0, x1, y1):
        self.coords = (x0, y0, x1, y1)
        
    def update_class_name(self, class_name):
        self.class_name = class_name

    def update_class_info(self, class_name, color):
        self.class_info_dict['class_name'] = class_name
        self.class_info_dict['color'] = color
        
    def make_x0y0_lesser_x1y1(self):
        # превращаем строки в числа и фиксируем координаты рамки, чтобы они не выходили за пределы кадра
        x0, y0, x1, y1 = self.coords
        x0 = np.clip(int(x0), 0, self.img_cols)
        x1 = np.clip(int(x1), 0, self.img_cols)
        y0 = np.clip(int(y0), 0, self.img_rows)
        y1 = np.clip(int(y1), 0, self.img_rows)

        # чтобы у нас ширина и высота рамки была не отрицательной, 
        # переставляем местами нулевую и первую координаты, если первая больше нулевой
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        self.coords = (x0, y0, x1, y1)

    def iou_compare(self, other_bbox, iou_threshold):
        iou = compute_iou(self.coords, other_bbox.coords)
        if iou <= iou_threshold:
            return True
        return False

    def __repr__(self) -> str:
        class_name = self.class_info_dict['class_name']
        return f'{class_name}: {self.coords}, {self.is_visible}'


def init_imshow_window(window_name, callback_function):
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_function)

def run_simple_image():
    img  = cv2.imread('aaa.jpg')
    frame_with_boxes = BboxFrame(img)
    window_name = 'image'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, frame_with_boxes)

    while True:
        img_with_boxes = frame_with_boxes.render_boxes()
        cv2.imshow(window_name, img_with_boxes)
        key = cv2.waitKey(0)          

        if key & 0xFF == 27:
            boxes_list = frame_with_boxes.bboxes_list
            break
    
    cv2.destroyAllWindows()
    

def run_simple_video():
    cap = cv2.VideoCapture(r'I:\aggr\4LUoqxnyxlE(+)+\4LUoqxnyxlE(+).mp4')

    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    class_names_list = ['person00', 'person01', 'person02']

    frame_with_boxes = BboxFrame(np.zeros((rows, cols, 3), dtype=np.uint8), class_names_list, class_names_list[1])

    window_name = 'Video'

    init_imshow_window(window_name, frame_with_boxes)

    while True:
        img_with_boxes = frame_with_boxes.render_boxes()
        cv2.imshow(window_name, img_with_boxes)
        
        key = cv2.waitKey(20)
        if key & 0xFF == 32:
            ret, frame = cap.read()
            if not ret:
                break
            frame_with_boxes.update_img(frame)

        if key & 0xFF == ord('n'):
            new_class_name = input('Enter the name of the new class:')
            frame_with_boxes.update_current_class_name(new_class_name)

        if key & 0xFF == ord('u'):
            frame_with_boxes.update_current_class_name('person00')

        if key & 0xFF == 27:
            boxes_list = frame_with_boxes.bboxes_list
            break
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    run_simple_video()
    
    