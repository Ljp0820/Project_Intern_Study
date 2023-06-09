import pygame
import numpy as np
import sys

# pygame 파라미터 선언
FPS = 10
clock=pygame.time.Clock()
clock.tick(FPS)

#size of window
screen_width=1200
screen_height=800

#color
WHITE=(255,255,255)
BLACK=(0,0,0)
GRAY=(127,127,127)

# pygame 초기화
pygame.init()
pygame.display.set_caption("Life Game")
display_surface=pygame.display.set_mode((screen_width, screen_height))
my_font_1=pygame.font.SysFont('malgungothic', 30)
my_font_2=pygame.font.SysFont('malgungothic', 14)

# 게임 초기화
field = np.zeros((100, 100))
cell_size=8
mode = "Edit"
time_step=0

def draw_surface(): # 기본화면 그리는 함수

    display_surface.fill(WHITE)
    
    for i in range(100):
        for j in range(100):
            if field[i,j] == 1:
                pygame.draw.rect(display_surface, BLACK, [i*cell_size, j*cell_size, cell_size, cell_size])

    for num_of_col in range(101): # 800*800 화면에 cell 크기가 8이므로 100번 반복
        for num_of_row in range(101):
            pygame.draw.line(display_surface, GRAY, [cell_size*num_of_col,0], [cell_size*num_of_col, 800]) 
            pygame.draw.line(display_surface, GRAY, [0,cell_size*num_of_row], [800, cell_size*num_of_row])
        
    text_1=my_font_1.render("Time Stop : " + str(time_step), True, BLACK)
    text_2=my_font_1.render("Mode : " + str(mode), True, BLACK)
    text_3=my_font_2.render("스페이스바로 실행합니다.", True, BLACK)
    text_4=my_font_2.render("공간안에서 마우스 우클릭시 블록을 지정할 수 있습니다.", True, BLACK)
    text_5=my_font_2.render("공간 안에서 마우스 좌클릭시 수정이 가능합니다.", True, BLACK)
    text_6=my_font_2.render("공간 밖에서 마우스 우클릭시 초기화됩니다.", True, BLACK)
    display_surface.blit(text_1, (820,10))
    display_surface.blit(text_2, (820,50))
    display_surface.blit(text_3, (810,700))
    display_surface.blit(text_4, (810,720))
    display_surface.blit(text_5, (810,740))
    display_surface.blit(text_6, (810,760))

def get_x_y(x):
    global field
    a,b=x
    a=a//8
    b=b//8
    field[a,b]=1

def del_x_y(x):
    global field
    a,b=x
    a=a//8
    b=b//8
    field[a,b]=0

while True:
    draw_surface()
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and event.button==1: # 좌클릭시 에딧모드
            if mode == "Edit":
                if pygame.mouse.get_pos()[0] < 801: # 800*800 화면 안에서 클릭할 경우에만
                    get_x_y(pygame.mouse.get_pos())
                else: # 800*800 화면 밖의 경우 전체 초기화
                    field=np.zeros((100,100))
                    time_step=0
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: #우클릭시 수정
            if pygame.mouse.get_pos()[0]<801:
                del_x_y(pygame.mouse.get_pos())
        elif event.type == pygame.KEYDOWN:
            if event.key==pygame.K_SPACE: # SPACE BAR로 러닝 모드 실행
                if mode == "Edit":
                    mode = "Run"
                else:
                    mode ="Edit"
        elif event.type == pygame.QUIT: # 창이 닫히는 이벤트가 발생하였는지
            pygame.quit()
            sys.exit()

    if mode == "Run":
        # Run에 맞는 논리
        time_step+=1

        field=np.pad(field, ((1,1), (1,1))) # padding array for slicing

        next_field=np.zeros((100, 100)) # 다음 time step의 정보 저장하게될 넘파이 어레이 100*100

        for i in range (1, 101): # row
            for j in range (1, 101): # col
                neighbor_array = field[i-1:i+2, j-1:j+2] # 3*3 matrix
                neighbor=np.sum(neighbor_array)-field[i,j] # 이웃의 수 = 3*3 행렬의 1의 갯수 - 본인
                if neighbor == 3: # 이웃 수 = 3 -> 생존 or 탄생
                    next_field[i-1,j-1]=1 # 원래 필드를 패딩했기 때문에 다음 필드의 인덱싱 시 행,열에 -1을 해야함
                    next_rect=[(i-1)*cell_size, (j-1)*cell_size, cell_size, cell_size]
                elif neighbor == 2: 
                    if field[i,j]==1: # 이웃이 2이고 현재 생존해 있는 경우 -> 생존
                        next_field[i-1,j-1]=1
                        next_rect=[(i-1)*cell_size, (j-1)*cell_size, cell_size, cell_size]
                    else: #이웃이 2이고 현재 없는 경우 -> 탄생 안함
                        next_field[i-1,j-1]=0
                else:# 이웃의 수가 1이하 or 4이상인 경우 무적권 없어짐
                    next_field[i-1,j-1]=0
        
        field=next_field

    pygame.display.update()
