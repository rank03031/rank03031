import tkinter as tk
from tkinter import filedialog, messagebox
from video_processor import process_video  # 비디오 처리 모듈 임포트


video_file_path = ""

def select_file():
    global video_file_path
    video_file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")))
    file_entry.config(state='normal')
    file_entry.delete(0, tk.END)
    file_entry.insert(0, video_file_path)
    file_entry.config(state='readonly')

def convert_video():
    if video_file_path:
        process_video(video_file_path)  # 변환
    else:
        messagebox.showwarning("경고", "비디오 파일을 선택하세요.")

def cancel():
    root.quit()

# GUI 설정
root = tk.Tk()
root.title("수화 번역기")

# 창 크기 설정
window_width = 600
window_height = 230
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")


center_frame = tk.Frame(root)
center_frame.pack(expand=True)

file_label = tk.Label(center_frame, text="파일:")
file_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')
file_entry = tk.Entry(center_frame, width=50, state='readonly', font=("Arial", 12))
file_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

# 파일 찾기 버튼
find_button = tk.Button(center_frame, text="파일 찾기", command=select_file, padx=10, pady=5)
find_button.grid(row=0, column=2, padx=10, pady=10)

# 변환 및 취소 버튼을 위한 프레임 생성
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, anchor='se', padx=10, pady=10)

convert_button = tk.Button(button_frame, text="변환", command=convert_video, padx=10, pady=5)
convert_button.grid(row=0, column=0, padx=10)
cancel_button = tk.Button(button_frame, text="닫기", command=cancel, padx=10, pady=5)
cancel_button.grid(row=0, column=1, padx=10)

center_frame.grid_columnconfigure(1, weight=1)

# GUI 창 실행
root.mainloop()
