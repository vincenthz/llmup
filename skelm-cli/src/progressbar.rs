use indicatif::ProgressStyle;
use skelm_download::ProgressDisplay;

pub struct ProgressBar(indicatif::ProgressBar);

impl ProgressDisplay for ProgressBar {
    fn progress_start(size: Option<u64>) -> Self {
        match size {
            Some(total) => {
                let pb = indicatif::ProgressBar::new(total);
                let progress_style = ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                             {bytes}/{total_bytes} ({eta})",
                )
                .unwrap();
                pb.set_style(progress_style);
                ProgressBar(pb)
            }
            None => {
                let pb = indicatif::ProgressBar::new_spinner();
                let progress_style = ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {bytes} downloaded",
                )
                .unwrap();
                pb.set_style(progress_style);
                ProgressBar(pb)
            }
        }
    }

    fn progress_update(&self, position: u64) {
        self.0.set_position(position)
    }

    fn progress_finalize(self) {
        self.0.finish()
    }
}
