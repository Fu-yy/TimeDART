



# draw3dpy310
class FourierSquareWave3D(ThreeDScene):
    from manim import *
    import numpy as np

    from manim import *
    import numpy as np
    def construct(self):
        # --- 1. 坐标轴（这里不 add，保持“只留波形”的效果） ---
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 6, 1],
            x_length=8,
            y_length=4,
            z_length=6
        )

        # --- 2. 相机角度 ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # --- 3. 参数 ---
        n_harmonics = 4  # 1,3,5,7
        colors = [BLUE_D, BLUE_C, BLUE_B, BLUE_A]

        # --- 4. 蓝色分量波形（往后排） ---
        for i in range(n_harmonics):
            n = 2 * i + 1
            amplitude = 1 / n
            z_depth = i + 1

            wave = ParametricFunction(
                lambda t, n=n, amp=amplitude, z=z_depth: axes.c2p(
                    t,
                    amp * np.sin(n * t),
                    z
                ),
                t_range=[-2 * np.pi, 2 * np.pi],
                color=colors[i % len(colors)],
                stroke_width=2
            )
            self.add(wave)

        # --- 5. 红色合成波（最前面） ---
        def square_approximation(t):
            y = 0.0
            for i in range(n_harmonics):
                n = 2 * i + 1
                y += (1 / n) * np.sin(n * t)
            return y

        sum_wave = ParametricFunction(
            lambda t: axes.c2p(
                t,
                square_approximation(t),
                0
            ),
            t_range=[-2 * np.pi, 2 * np.pi],
            color=RED,
            stroke_width=4
        )
        self.add(sum_wave)

        # --- 6. 不做旋转，输出静态图 ---
        # self.begin_ambient_camera_rotation(rate=0.2)
        # self.wait(4)

        # 关键：给渲染器一个“落帧”的时刻，确保能保存 last frame
        self.wait(0.1)


if __name__ == '__main__':
    # draw_fourier()
    # ====== 关键配置：只要图片，不要视频 ======
    config.verbosity = "WARNING"

    # 输出图片（保存最后一帧）
    config.save_last_frame = True
    config.write_to_movie = False
    config.format = "png"

    # 分辨率（可按需改）
    config.pixel_width = 1600
    config.pixel_height = 900

    # 文件名（不含扩展名也行）
    config.output_file = "fourier_square_3d"

    # 直接渲染
    scene = FourierSquareWave3D()
    scene.render()
