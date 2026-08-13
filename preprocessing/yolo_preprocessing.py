import os,json
import shutil

#json파일 기준으로 라벨 추출
def label_from_json(data_list):
    # i=> 한 개의 라벨
    #'Code Name': 'A220120XX_10337.jpg' -> .을 기준으로 나눠서 앞부분 가져옴 -> A220120XX_10337
    filename = data_list['Code Name']


    #너비, 높이 추출
    w = data_list['W']
    h = data_list['H']


    #x, y 센터 포인트
    x, y = data_list['Point(x,y)'].split(',')


    w = float(w)
    h = float(h)
    x = float(x)
    y = float(y)


    # print(f'{filename}에서 추출된 대상 : {x}, {y}, {w}, {h}')
    return x, y, w, h, filename


#txt파일 기준으로 라벨 추출
def label_from_txt(sample_path):
    with open(sample_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = []


        # ['Code', 'Name', 'A220120XX10306.jpg']
        # ['Pointxy', '0.317078189300412, 0.479356405585914']
        # ['W', '0.633333333333333'], ['H', '0.957498482088646']
        for line in lines :
            #공백제거
            parts = line.strip().split()
            words.append([re.sub(r'[^a-zA-Z0-9.,]', '', x) for x in parts])


        Width, Height = 0, 0
        point_x, point_y = 0, 0
        path=''
        for w in words:
            if 'W' in w:
                Width = w[1]
            if 'H' in w:
                Height = w[1]
            if 'Pointx,y' in w:
                point_x, point_y = w[1].split(',')[0], w[1].split(',')[1]
            if 'Code' in w:
                path = w[2]
        print(Width, Height, point_x, point_y, path)

def create_yolo_label(label_folder):
    sample_label_path = r'Data\PeachDataset\peach_label\A220120XX_10316.json'
    

    # image_folder = r'./Data/PeachDataset/peach_image'
    # label_folder = r'./Data/PeachDataset/peach_label'
    
    json_list = [os.path.join(label_folder, x) for x in os.listdir(label_folder) if 'json' in x]

    #여러 JSON 라벨 파일을 YOLO가 읽을 수 있는 .txt 라벨 파일로 변환하기 위해 준비하는 과정
    for i in range(len(json_list)):
        with open(json_list[i], 'r', encoding='utf-8') as f:
            data_list = json.load(f)

            lines=[]
            #data ->한개의 json파일 안에 있는 한개의 라벨
            for data in data_list:
                #하나의 라벨 덩어리에서 x,y,w,h,filename을 추출
                x, y, w, h, filename = label_from_json(data)

                #txt파일로 변환!
                lines.append(f'0   {x}   {y}   {w}   {h}\n')
            out_path = os.path.join(label_folder,'yolo_txt_label')
            # 경로가 없을때 -> os.mkdir(경로) 만들어줘!
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            
            txt_path = f'{out_path}\{filename}.txt'
            print(txt_path)
            #'파일이름'으로 lines 리스트를 txt파일로 저장
            with open(txt_path,'w',encoding='utf-8') as f:
                f.writelines(lines)

def create_data_directory(base_dir):
    #base_dir/images/train
    image = os.path.join(base_dir,'images')
    label = os.path.join(base_dir,'labels')
    image_train = os.path.join(base_dir,'images','train')
    image_valid = os.path.join(base_dir,'images','valid')
    label_train = os.path.join(base_dir,'images','train')
    label_valid = os.path.join(base_dir,'images','valid')

    for p in [image,label,image_train,image_valid,label_train,label_valid]:
        #p라는 경로의 폴더가 있는지 확인
        if not os.path.exists(p):
            os.mkdir(p)
            print(f'{p} 경로를 생성하였음.')


def move_label_datas(
    source_label_folder,
    train_image_folder,
    valid_image_folder,
    train_label_folder,
    valid_label_folder,
):
    source_label_paths = [
        os.path.join(source_label_folder, filename)
        for filename in os.listdir(source_label_folder)
        if filename.endswith('.txt')
    ]

    train_image_names = set(os.listdir(train_image_folder))
    valid_image_names = set(os.listdir(valid_image_folder))

    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(valid_label_folder, exist_ok=True)

    for source_label_path in source_label_paths:
        source_label_name = os.path.basename(source_label_path)

        # A220120XX_10317.jpg.txt → A220120XX_10317.txt
        image_stem = os.path.splitext(
            os.path.splitext(source_label_name)[0]
        )[0]

        image_name = image_stem + '.jpg'
        yolo_label_name = image_stem + '.txt'

        if image_name in train_image_names:
            destination_path = os.path.join(train_label_folder, yolo_label_name)
            shutil.copy2(source_label_path, destination_path)

        elif image_name in valid_image_names:
            destination_path = os.path.join(valid_label_folder, yolo_label_name)
            shutil.copy2(source_label_path, destination_path)