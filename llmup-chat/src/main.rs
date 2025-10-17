use std::borrow::Cow;

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
use rust_embed::RustEmbed;

use crate::title::AppTitleBar;

mod themes;
mod title;

/// An asset source that loads assets from the `./assets` folder.
#[derive(RustEmbed)]
#[folder = "./assets"]
#[include = "icons/**/*.svg"]
pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if path.is_empty() {
            return Ok(None);
        }

        Self::get(path)
            .map(|f| Some(f.data))
            .ok_or_else(|| anyhow::anyhow!("could not find asset at path \"{path}\""))
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(Self::iter()
            .filter_map(|p| p.starts_with(path).then(|| p.into()))
            .collect())
    }
}

#[derive(Action, Clone, PartialEq, Eq)]
#[action(namespace = story, no_json)]
pub struct SelectLocale(SharedString);

actions!(llmupchat, [About, Quit, CloseWindow, OpenWebsite]);

pub struct History {
    pub title: String,
    pub content: String,
}

pub struct LLMChat {
    title_bar: Entity<AppTitleBar>,
    history: Vec<Entity<History>>,
    sidebar_state: Entity<ResizableState>,
    search_input: Entity<InputState>,
    collapsed: bool,
    input1: Entity<InputState>,
}

impl LLMChat {
    pub fn new(cx: &mut Context<Self>, window: &mut Window) -> Self {
        let hist = History {
            title: String::from("some chat"),
            content: String::from("this is the content of the chat"),
        };
        let hist = cx.new(|_| hist);
        let search_input = cx.new(|cx| InputState::new(window, cx).placeholder("Search..."));
        let title_bar = cx.new(|cx| AppTitleBar::new("LlmChat", window, cx));

        Self {
            title_bar,
            history: vec![hist],
            sidebar_state: ResizableState::new(cx),
            collapsed: false,
            input1: cx.new(|cx| InputState::new(window, cx)),
            search_input,
        }
    }
}

#[derive(IntoElement)]
struct StorySection {
    base: Div,
    title: SharedString,
    sub_title: Vec<AnyElement>,
    children: Vec<AnyElement>,
}

impl StorySection {
    pub fn sub_title(mut self, sub_title: impl IntoElement) -> Self {
        self.sub_title.push(sub_title.into_any_element());
        self
    }

    #[allow(unused)]
    fn max_w_md(mut self) -> Self {
        self.base = self.base.max_w(rems(48.));
        self
    }

    #[allow(unused)]
    fn max_w_lg(mut self) -> Self {
        self.base = self.base.max_w(rems(64.));
        self
    }

    #[allow(unused)]
    fn max_w_xl(mut self) -> Self {
        self.base = self.base.max_w(rems(80.));
        self
    }

    #[allow(unused)]
    fn max_w_2xl(mut self) -> Self {
        self.base = self.base.max_w(rems(96.));
        self
    }
}

impl ParentElement for StorySection {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements);
    }
}

impl Styled for StorySection {
    fn style(&mut self) -> &mut gpui::StyleRefinement {
        self.base.style()
    }
}

impl RenderOnce for StorySection {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        GroupBox::new()
            .outline()
            .title(
                h_flex()
                    .justify_between()
                    .w_full()
                    .gap_4()
                    .child(self.title)
                    .children(self.sub_title),
            )
            .content_style(
                StyleRefinement::default()
                    .rounded_lg()
                    .overflow_x_hidden()
                    .items_center()
                    .justify_center(),
            )
            .child(self.base.children(self.children))
    }
}

impl ContextMenuExt for StorySection {}

pub(crate) fn section(title: impl Into<SharedString>) -> StorySection {
    StorySection {
        title: title.into(),
        sub_title: vec![],
        base: h_flex()
            .flex_wrap()
            .justify_center()
            .items_center()
            .w_full()
            .gap_4(),
        children: vec![],
    }
}

pub(crate) fn user_input(content: impl Into<SharedString>) -> impl IntoElement {
    GroupBox::new().bg(green_400()).child(Label::new(content))
}

impl Render for LLMChat {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.search_input.read(cx).value().trim().to_lowercase();

        let histories = self.history.iter().collect::<Vec<_>>();

        div().size_full().child(
            v_flex().size_full().child(self.title_bar.clone()).child(
                h_resizable("llmchat-container", self.sidebar_state.clone())
                    .child(
                        resizable_panel()
                            .size(px(255.))
                            .size_range(px(200.)..px(320.))
                            .child(
                                Sidebar::left()
                                    .width(relative(1.))
                                    .border_width(px(0.))
                                    .collapsed(self.collapsed)
                                    .header(
                                        v_flex()
                                            .w_full()
                                            .gap_4()
                                            .child(
                                                SidebarHeader::new()
                                                    .w_full()
                                                    .child(Label::new("History")),
                                            )
                                            .child(
                                                div()
                                                    .bg(cx.theme().sidebar_accent)
                                                    .px_1()
                                                    .rounded_full()
                                                    .flex_1()
                                                    .mx_1()
                                                    .child(
                                                        TextInput::new(&self.search_input)
                                                            .appearance(false)
                                                            .cleanable(),
                                                    ),
                                            ),
                                    )
                                    .child(SidebarMenu::new().children(
                                        histories.clone().into_iter().enumerate().map(
                                            |(idx, history)| {
                                                //SidebarMenu::menu().children()
                                                SidebarMenuItem::new(history.read(cx).title.clone())
                                            },
                                        ),
                                    )),
                            ),
                    )
                    .child(
                        v_flex()
                            .flex_1()
                            .h_full()
                            .overflow_x_hidden()
                            .child(user_input("Tell me a story about rust"))
                            .child(
                                GroupBox::new().outline().border_1().child(
                                    div().bg(blue_400()).w(px(200.)).child(
                                        Label::new(
                                            "Label should support text wrap in default, \
                                        if the text is too long, it should wrap to the next line.",
                                        )
                                        .line_height(rems(1.8)),
                                    ),
                                ),
                            )
                            .child(
                                section("Input")
                                    .max_w_md()
                                    .child(TextInput::new(&self.input1).cleanable()),
                            )
                            .into_any_element(),
                    ),
            ),
        )
    }
}

fn init(cx: &mut App) {
    gpui_component::init(cx);
    themes::init(cx);
    cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);

    cx.on_action(|_: &Quit, cx: &mut App| {
        cx.quit();
    });

    cx.activate(true);
}

fn main() {
    let app = Application::new().with_assets(Assets);

    let title = SharedString::from("LLMChat".to_string());
    app.run(move |cx| {
        init(cx);

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
