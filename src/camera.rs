use crate::math::Vector3;

pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
    pub eta_ratio: f64,
}

impl Ray {
    pub fn new(origin: Vector3, direction: Vector3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            eta_ratio: 1.0,
        }
    }

    pub fn at(&self, t: f64) -> Vector3 {
        self.origin + self.direction * t
    }

    pub fn set_eta_ratio(&mut self, eta_ratio: f64) {
        self.eta_ratio = eta_ratio;
    }
}

pub struct Camera {
    origin: Vector3,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    screen_distance: f64,
    #[allow(dead_code)]
    fov_horizontal: f64, // 水平視野角（ラジアン）
    // キャッシュされた計算結果
    lower_left_corner: Vector3,
    horizontal: Vector3,
    vertical: Vector3,
}

impl Camera {
    /// カメラを作成
    ///
    /// # Arguments
    /// * `origin` - カメラの位置
    /// * `width` - 画面の幅（ピクセル）
    /// * `height` - 画面の高さ（ピクセル）
    /// * `screen_distance` - スクリーンとカメラの距離
    /// * `fov_degrees` - 水平視野角（度）
    pub fn new(
        origin: Vector3,
        width: u32,
        height: u32,
        screen_distance: f64,
        fov_degrees: f64,
    ) -> Self {
        let fov_horizontal = fov_degrees.to_radians();
        let aspect_ratio = width as f64 / height as f64;

        // 視野角からスクリーンの幅と高さを計算
        let viewport_height = 2.0 * (fov_horizontal / 2.0).tan() * screen_distance;
        let viewport_width = viewport_height * aspect_ratio;

        // カメラの向きベクトル（デフォルトはZ-方向を見る）
        let horizontal = Vector3::new(viewport_width, 0.0, 0.0);
        let vertical = Vector3::new(0.0, viewport_height, 0.0);

        // スクリーンの左下隅の位置
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - Vector3::new(0.0, 0.0, screen_distance);

        Self {
            origin,
            width,
            height,
            screen_distance,
            fov_horizontal,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }

    /// カメラの向きを設定（look_at）
    ///
    /// # Arguments
    /// * `origin` - カメラの位置
    /// * `target` - 見る対象の位置
    /// * `up` - 上方向ベクトル
    /// * `width` - 画面の幅（ピクセル）
    /// * `height` - 画面の高さ（ピクセル）
    /// * `fov_degrees` - 水平視野角（度）
    #[allow(dead_code)]
    pub fn look_at(
        origin: Vector3,
        target: Vector3,
        up: Vector3,
        width: u32,
        height: u32,
        fov_degrees: f64,
    ) -> Self {
        let fov_horizontal = fov_degrees.to_radians();
        let aspect_ratio = width as f64 / height as f64;

        // カメラの座標系を構築
        let w = (origin - target).normalize(); // カメラの後ろ方向
        let u = up.cross(&w).normalize(); // カメラの右方向
        let v = w.cross(&u); // カメラの上方向

        // スクリーンの距離を1.0とした場合のビューポートサイズ
        let screen_distance = 1.0;
        let viewport_height = 2.0 * (fov_horizontal / 2.0).tan() * screen_distance;
        let viewport_width = viewport_height * aspect_ratio;

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;

        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w * screen_distance;

        Self {
            origin,
            width,
            height,
            screen_distance,
            fov_horizontal,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }

    /// アンチエイリアシング用：ピクセル内のランダムな位置のレイを生成
    ///
    /// # Arguments
    /// * `x` - ピクセルのX座標
    /// * `y` - ピクセルのY座標
    /// * `offset_x` - ピクセル内のオフセット (0.0 ~ 1.0)
    /// * `offset_y` - ピクセル内のオフセット (0.0 ~ 1.0)
    pub fn get_ray_with_offset(&self, x: u32, y: u32, offset_x: f64, offset_y: f64) -> Ray {
        let u = (x as f64 + offset_x) / (self.width - 1) as f64;
        let v = (y as f64 + offset_y) / (self.height - 1) as f64;

        let direction =
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin;

        Ray::new(self.origin, direction)
    }
}
