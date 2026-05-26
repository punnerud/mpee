use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("http: {0}")]
    Http(String),

    #[error("invalid input: {0}")]
    Invalid(String),

    #[error("infeasible: {0}")]
    Infeasible(String),

    #[error("matrix: {0}")]
    Matrix(String),

    #[error("{0}")]
    Other(String),
}

#[cfg(feature = "osrm")]
impl From<ureq::Error> for Error {
    fn from(value: ureq::Error) -> Self {
        Error::Http(value.to_string())
    }
}
