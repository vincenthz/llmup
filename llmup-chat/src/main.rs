use gpui::*;
use gpui_component::{
    ActiveTheme, IconName, Root, TitleBar, blue_400,
    button::{Button, ButtonVariants},
    context_menu::ContextMenuExt,
    green_400,
    group_box::GroupBox,
    h_flex,
    input::{InputState, TextInput},
    label::Label,
    resizable::{ResizableState, h_resizable, resizable_panel},
    sidebar::{Sidebar, SidebarHeader, SidebarMenu, SidebarMenuItem},
    v_flex,
};
use llmup_chat::{AppSidebar, AppTitleBar, Playground};

pub struct LLMChat {
    title_bar: Entity<AppTitleBar>,
    app_sidebar: Entity<AppSidebar>,
    playground: Entity<Playground>,
}

impl LLMChat {
    pub fn new(cx: &mut Context<Self>, window: &mut Window) -> Self {
        let title_bar = cx.new(|cx| AppTitleBar::new("LlmChat", window, cx));
        let app_sidebar = cx.new(|cx| AppSidebar::new(window, cx));
        let playground = cx.new(|cx| Playground::new(window, cx));

        Self {
            title_bar,
            app_sidebar,
            playground,
        }
    }
}

impl Render for LLMChat {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        /*
        match self.app_sidebar.read(cx).active() {
            crate::sidebar::Item::Playground => (),
        };
        */
        div().size_full().child(
            v_flex().size_full().child(self.title_bar.clone()).child(
                h_flex()
                    .border_1()
                    .border_color(cx.theme().border)
                    .h_full()
                    .child(self.app_sidebar.clone())
                    .child(
                        div()
                            .id("main")
                            .flex_1()
                            .size_full()
                            .child(self.playground.clone()),
                    ),
                //.child(div().id("main").flex_1().child(self.active[]))
                //.child(v_flex().size_full().gap_4().p_4().child(Label::new("a"))),
            ),
        )
        /*

        div().size_full().child(
            v_flex().size_full().child(self.title_bar.clone()).child(
            ),
        )
        */
    }
}

fn main() {
    let app = Application::new().with_assets(llmup_chat::Assets);

    let title = SharedString::from("LLMChat".to_string());
    app.run(move |cx| {
        llmup_chat::init(cx);

        cx.spawn(async move |cx| {
            let window_options = WindowOptions {
                titlebar: Some(TitleBar::title_bar_options()),
                kind: WindowKind::Normal,
                #[cfg(target_os = "linux")]
                window_decorations: Some(gpui::WindowDecorations::Client),
                ..Default::default()
            };

            let window = cx.open_window(window_options, |window, cx| {
                let view = cx.new(|ctx| LLMChat::new(ctx, window));
                cx.new(|cx| Root::new(view.into(), window, cx))
            })?;

            window.update(cx, |_, window, _| {
                window.activate_window();
                window.set_window_title(&title);
            })?;

            Ok::<_, anyhow::Error>(())
        })
        .detach();
    });
}
