CREATE DATABASE quanlysach
USE quanlysach;
CREATE TABLE loai_sach (
    ma_loai CHAR(5) PRIMARY KEY,
    ten_loai NVARCHAR(100) NOT NULL
);
CREATE TABLE sach (
    ma_sach NVARCHAR(20) PRIMARY KEY,
    ma_loai CHAR(5) NOT NULL,
    nhan_de NVARCHAR(255) NOT NULL,
    tac_gia NVARCHAR(150) NOT NULL,
    nha_xuat_ban NVARCHAR(150) NOT NULL,
    mo_ta TEXT,
    hinh_anh NVARCHAR(255),
    so_trang INT NOT NULL,
    gia_bia DECIMAL(12,2) NOT NULL,
    so_luong INT NOT NULL DEFAULT 0,

    CONSTRAINT fk_sach_loai
        FOREIGN KEY (ma_loai)
        REFERENCES loai_sach(ma_loai),

    CONSTRAINT chk_sach_so_trang
        CHECK (so_trang > 0),

    CONSTRAINT chk_sach_gia_bia
        CHECK (gia_bia >= 0),

    CONSTRAINT chk_sach_so_luong
        CHECK (so_luong >= 0)
);
----
INSERT INTO loai_sach (ma_loai, ten_loai)
VALUES
('L0001', N'Văn học'),
('L0002', N'Du lịch'),
('L0003', N'Xây dựng'),
('L0004', N'Kinh doanh'),
('L0005', N'Công nghệ thông tin');

INSERT INTO sach
(ma_sach, ma_loai, nhan_de, tac_gia,
 nha_xuat_ban, mo_ta, hinh_anh,
 so_trang, gia_bia, so_luong)
VALUES

('S001', 'L0001', N'Dế Mèn phiêu lưu ký',
 N'Tô Hoài', N'Nhà xuất bản Kim Đồng',
 N'Tác phẩm văn học thiếu nhi nổi tiếng của Việt Nam.',
 'demen.jpg', 200, 80000, 30),

('S002', 'L0001', N'Nhà giả kim',
 N'Paulo Coelho', N'Nhà xuất bản Hội Nhà Văn',
 N'Tiểu thuyết kể về hành trình theo đuổi ước mơ.',
 'nhagiakim.jpg', 228, 95000, 15),

('S003', 'L0004', N'Đắc nhân tâm',
 N'Dale Carnegie', N'Nhà xuất bản Tổng hợp',
 N'Cuốn sách về nghệ thuật giao tiếp và ứng xử.',
 'dacnhantam.jpg', 320, 110000, 25),

('S004', 'L0005', N'Lập trình Python cơ bản',
 N'Nguyễn Văn An', N'Nhà xuất bản Giáo dục',
 N'Giới thiệu kiến thức lập trình Python dành cho người mới.',
 'python.jpg', 250, 120000, 20),

('S005', 'L0005', N'Giáo trình Cơ sở dữ liệu',
 N'Trần Văn Bình', N'Nhà xuất bản Giáo dục',
 N'Cung cấp kiến thức nền tảng về hệ quản trị cơ sở dữ liệu.',
 'csdl.jpg', 350, 150000, 10);


SELECT * FROM loai_sach;

SELECT * FROM sach;
