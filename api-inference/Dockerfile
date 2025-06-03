# Usa una imagen base de Python
FROM python:3.9

#------------------------------------INSTALL UV---------------------------------------
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"
#-------------------------------------------------------------------------------------

# Instala las dependencias de Python desde requirements.txt
RUN uv init api

WORKDIR api

# Copia todos los archivos del proyecto dentro del contenedor
COPY . .

RUN ["uv","add","-r","requirements.txt"]

#CMD ["cat", "pyproject.toml"]
CMD ["ls", "-l"]

