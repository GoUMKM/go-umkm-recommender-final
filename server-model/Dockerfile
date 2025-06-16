# Gunakan image Nginx resmi sebagai base image
FROM nginx:stable-alpine

# Buat direktori kerja dalam container
WORKDIR /app

# Salin folder artifacts ke lokasi yang sesuai dengan konfigurasi Nginx
COPY artifacts /app/artifacts

# Salin konfigurasi Nginx ke dalam container
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80 agar dapat diakses secara publik
EXPOSE 80

# Jalankan Nginx secara default
CMD ["nginx", "-g", "daemon off;"]